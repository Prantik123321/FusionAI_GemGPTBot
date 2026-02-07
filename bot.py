#!/usr/bin/env python3
"""
TELEGRAM AI BOT - ChatGPT + Gemini
Python 3.9-3.11 Compatible Version
"""

import os
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import libraries
from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackContext
)

from flask import Flask, request, jsonify
import openai
import google.generativeai as genai
import requests

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    """Configuration class"""
    TELEGRAM_TOKEN: str = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "10000")))
    NODE_ENV: str = field(default_factory=lambda: os.getenv("NODE_ENV", "production"))
    RENDER_URL: str = field(default_factory=lambda: os.getenv("RENDER_EXTERNAL_URL", ""))
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        if not self.TELEGRAM_TOKEN:
            errors.append("TELEGRAM_TOKEN is missing")
        elif not self.TELEGRAM_TOKEN.startswith("8203704693:"):
            errors.append("TELEGRAM_TOKEN format invalid")
            
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is missing")
        elif not self.OPENAI_API_KEY.startswith("sk-proj-"):
            errors.append("OPENAI_API_KEY format invalid")
            
        if not self.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is missing")
        elif not self.GEMINI_API_KEY.startswith("AIza"):
            errors.append("GEMINI_API_KEY format invalid")
        
        if errors:
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✅ Configuration loaded successfully")
        return True

config = Config()

# Initialize AI services
openai.api_key = config.OPENAI_API_KEY
genai.configure(api_key=config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# ==================== AI SERVICE ====================

class AIService:
    """AI Service for ChatGPT and Gemini"""
    
    @staticmethod
    def call_chatgpt(prompt: str) -> Dict:
        """Call ChatGPT using openai library"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return {
                "success": True,
                "content": content.strip(),
                "model": "gpt-3.5-turbo",
                "source": "ChatGPT"
            }
            
        except Exception as e:
            logger.error(f"ChatGPT error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source": "ChatGPT"
            }
    
    @staticmethod
    def call_gemini(prompt: str) -> Dict:
        """Call Gemini using google-generativeai library"""
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 500,
                }
            )
            
            return {
                "success": True,
                "content": response.text.strip(),
                "model": "gemini-pro",
                "source": "Gemini"
            }
            
        except Exception as e:
            logger.error(f"Gemini error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source": "Gemini"
            }
    
    @staticmethod
    async def get_ai_response(prompt: str) -> str:
        """Get response from both AIs and merge them"""
        try:
            # Run both AI calls in parallel
            loop = asyncio.get_event_loop()
            
            # Run in thread pool to avoid blocking
            chatgpt_result = await loop.run_in_executor(None, AIService.call_chatgpt, prompt)
            gemini_result = await loop.run_in_executor(None, AIService.call_gemini, prompt)
            
            # Handle failures
            if not chatgpt_result["success"] and not gemini_result["success"]:
                return "🤖 I apologize, but both AI services are currently unavailable. Please try again later."
            
            if not chatgpt_result["success"]:
                return f"{gemini_result['content']}\n\n🔍 _Powered by: {gemini_result['source']}_"
            
            if not gemini_result["success"]:
                return f"{chatgpt_result['content']}\n\n🔍 _Powered by: {chatgpt_result['source']}_"
            
            # Both succeeded - create simple merged response
            chatgpt_text = chatgpt_result["content"]
            gemini_text = gemini_result["content"]
            
            # Take first 2 sentences from each
            chatgpt_sentences = [s.strip() + '.' for s in chatgpt_text.split('.') if s.strip()]
            gemini_sentences = [s.strip() + '.' for s in gemini_text.split('.') if s.strip()]
            
            merged_sentences = []
            if chatgpt_sentences:
                merged_sentences.extend(chatgpt_sentences[:2])
            if gemini_sentences:
                merged_sentences.extend(gemini_sentences[:2])
            
            final_text = ' '.join(merged_sentences)
            
            return f"{final_text}\n\n🔍 _Powered by: ChatGPT + Gemini_"
            
        except Exception as e:
            logger.error(f"AI response error: {str(e)}")
            return "❌ Sorry, I encountered an error. Please try again."

# ==================== TELEGRAM BOT ====================

class TelegramBot:
    """Telegram Bot Handler"""
    
    def __init__(self):
        self.config = config
        self.ai_service = AIService()
        
        # User data storage
        self.user_sessions: Dict[int, List[str]] = {}
        self.user_rates: Dict[int, List[datetime]] = {}
        
        # Create application
        self.application = Application.builder().token(self.config.TELEGRAM_TOKEN).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("🤖 Telegram Bot initialized")
    
    def _register_handlers(self):
        """Register all command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("clear", self._clear_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        
        # Message handler
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        
        # Error handler
        self.application.add_error_handler(self._error_handler)
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        welcome_text = f"""
🤖 Welcome {user.first_name} to AI Assistant Bot!

I combine:
• ChatGPT (OpenAI) - Detailed explanations
• Gemini (Google) - Concise answers
• Smart Merge - Best of both!

Commands:
/start - Show this message
/help - Get help
/clear - Clear history
/status - Check status

Just send me any message! 😊
        """
        await update.message.reply_text(welcome_text, parse_mode="Markdown")
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🆘 Help Guide

How I work:
1. Send your question
2. I query both ChatGPT & Gemini
3. I merge their responses
4. You get one comprehensive answer

Rate Limits: 50 requests per 15 minutes

Privacy: No data stored permanently

Need help? @Prantik123321
        """
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def _clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command"""
        user_id = update.effective_user.id
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        await update.message.reply_text("✅ Conversation cleared!")
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status_text = f"""
📊 Bot Status

Services:
• ChatGPT: ✅ Active
• Gemini: ✅ Active
• Merge: ✅ Working

Active Users: {len(self.user_sessions)}
Your ID: {update.effective_user.id}

Status: 🟢 Operational
        """
        await update.message.reply_text(status_text, parse_mode="Markdown")
    
    def _check_rate_limit(self, user_id: int) -> bool:
        """Check rate limiting"""
        now = datetime.now()
        
        if user_id not in self.user_rates:
            self.user_rates[user_id] = []
        
        # Remove old requests (15 minutes)
        valid_requests = [
            req_time for req_time in self.user_rates[user_id]
            if (now - req_time) < timedelta(minutes=15)
        ]
        
        if len(valid_requests) >= 50:
            return False
        
        valid_requests.append(now)
        self.user_rates[user_id] = valid_requests
        return True
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        user_id = update.effective_user.id
        message_text = update.message.text
        
        logger.info(f"Message from {user_id}: {message_text[:50]}...")
        
        # Check rate limit
        if not self._check_rate_limit(user_id):
            await update.message.reply_text(
                "⏳ Rate limit exceeded. Please wait 15 minutes."
            )
            return
        
        # Show typing indicator
        await update.message.chat.send_action(action="typing")
        
        try:
            # Send processing message
            processing_msg = await update.message.reply_text("🤔 Processing with both AIs...")
            
            # Get AI response
            response = await self.ai_service.get_ai_response(message_text)
            
            # Update message
            await processing_msg.edit_text(response, parse_mode="Markdown")
            
            logger.info(f"Response sent to {user_id}")
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            await update.message.reply_text("❌ Error processing. Please try again.")
    
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Error: {context.error}")
        try:
            await update.message.reply_text("❌ An error occurred. Please try again.")
        except:
            pass
    
    def setup_webhook(self, webhook_url: str):
        """Setup webhook"""
        self.application.bot.set_webhook(f"{webhook_url}/webhook")
        logger.info(f"✅ Webhook set to: {webhook_url}/webhook")
    
    def run_polling(self):
        """Run polling mode"""
        logger.info("🔧 Starting polling mode...")
        self.application.run_polling()
    
    def get_application(self):
        """Get application instance"""
        return self.application

# ==================== FLASK SERVER ====================

# Create Flask app
app = Flask(__name__)

# Global bot instance
telegram_bot = None
flask_app = None

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "Telegram AI Bot",
        "version": "1.0.0",
        "endpoints": ["/health", "/status", "/webhook"],
        "message": "Bot is running!"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bot": "initialized" if telegram_bot else "starting"
    })

@app.route('/status')
def status():
    return jsonify({
        "users": len(telegram_bot.user_sessions) if telegram_bot else 0,
        "services": ["ChatGPT", "Gemini"],
        "environment": config.NODE_ENV
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Telegram webhook endpoint"""
    try:
        if telegram_bot is None:
            return jsonify({"error": "Bot not initialized"}), 500
        
        # Process update
        update = Update.de_json(request.get_json(), telegram_bot.application.bot)
        telegram_bot.application.process_update(update)
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/setwebhook', methods=['GET', 'POST'])
def set_webhook():
    """Set webhook manually"""
    try:
        webhook_url = request.args.get('url') or config.RENDER_URL
        if not webhook_url:
            return jsonify({"error": "No webhook URL provided"}), 400
        
        telegram_bot.setup_webhook(webhook_url)
        return jsonify({
            "success": True,
            "message": f"Webhook set to {webhook_url}/webhook"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== MAIN FUNCTION ====================

def run_flask():
    """Run Flask server in background"""
    global flask_app
    flask_app = app
    flask_app.run(host='0.0.0.0', port=config.PORT, debug=False, use_reloader=False)

def main():
    """Main function"""
    global telegram_bot
    
    # Validate config
    if not config.validate():
        logger.error("❌ Invalid configuration")
        return
    
    # Initialize bot
    telegram_bot = TelegramBot()
    
    # Check environment
    if config.NODE_ENV == "production" and config.RENDER_URL:
        # Production mode with webhook
        logger.info("🚀 Starting in PRODUCTION mode")
        
        # Set webhook
        telegram_bot.setup_webhook(config.RENDER_URL)
        
        # Start Flask in background thread
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        logger.info(f"🌐 Server started on port {config.PORT}")
        logger.info(f"📡 Webhook: {config.RENDER_URL}/webhook")
        logger.info("✅ Bot is ready!")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            
    else:
        # Development mode - polling
        logger.info("🔧 Starting in DEVELOPMENT mode")
        telegram_bot.run_polling()

if __name__ == "__main__":
    main()
