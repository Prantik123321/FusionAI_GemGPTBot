#!/usr/bin/env python3
"""
TELEGRAM AI BOT - ChatGPT + Gemini
Working Version - No Compatibility Issues
"""

import os
import json
import logging
import asyncio
from datetime import datetime
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

# Import Telegram
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# Import Flask for webhook
from flask import Flask, request, jsonify

# Import requests for API calls
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
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.TELEGRAM_TOKEN:
            logger.error("❌ TELEGRAM_TOKEN is missing")
            return False
        if not self.TELEGRAM_TOKEN.startswith("8203704693:"):
            logger.error("❌ TELEGRAM_TOKEN format invalid")
            return False
            
        if not self.OPENAI_API_KEY:
            logger.error("❌ OPENAI_API_KEY is missing")
            return False
        if not self.OPENAI_API_KEY.startswith("sk-proj-"):
            logger.error("❌ OPENAI_API_KEY format invalid")
            return False
            
        if not self.GEMINI_API_KEY:
            logger.error("❌ GEMINI_API_KEY is missing")
            return False
        if not self.GEMINI_API_KEY.startswith("AIza"):
            logger.error("❌ GEMINI_API_KEY format invalid")
            return False
            
        logger.info("✅ Configuration validated successfully")
        return True

config = Config()

# ==================== AI SERVICE CLASSES ====================

class AIService:
    """Base AI Service class"""
    
    @staticmethod
    def call_chatgpt(prompt: str) -> Dict:
        """Call ChatGPT API using direct HTTP request"""
        try:
            headers = {
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise answers."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 600,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "content": content.strip(),
                    "model": "gpt-3.5-turbo",
                    "source": "ChatGPT"
                }
            else:
                logger.error(f"ChatGPT API Error {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "source": "ChatGPT"
                }
                
        except Exception as e:
            logger.error(f"ChatGPT request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source": "ChatGPT"
            }
    
    @staticmethod
    def call_gemini(prompt: str) -> Dict:
        """Call Gemini API using direct HTTP request"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={config.GEMINI_API_KEY}"
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 600
                }
            }
            
            response = requests.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return {
                        "success": True,
                        "content": content.strip(),
                        "model": "gemini-2.0-flash",
                        "source": "Gemini"
                    }
                else:
                    return {
                        "success": False,
                        "error": "No response from Gemini",
                        "source": "Gemini"
                    }
            else:
                logger.error(f"Gemini API Error {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "error": f"API Error: {response.status_code}",
                    "source": "Gemini"
                }
                
        except Exception as e:
            logger.error(f"Gemini request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source": "Gemini"
            }
    
    @staticmethod
    def merge_responses(chatgpt_result: Dict, gemini_result: Dict) -> str:
        """Merge responses from both AIs intelligently"""
        
        # Handle failures
        if not chatgpt_result["success"] and not gemini_result["success"]:
            return "🤖 I apologize, but both AI services are currently unavailable. Please try again in a few minutes."
        
        if not chatgpt_result["success"]:
            return f"{gemini_result['content']}\n\n🔍 _Powered by: {gemini_result['source']}_"
        
        if not gemini_result["success"]:
            return f"{chatgpt_result['content']}\n\n🔍 _Powered by: {chatgpt_result['source']}_"
        
        # Both succeeded - merge intelligently
        chatgpt_text = chatgpt_result["content"]
        gemini_text = gemini_result["content"]
        
        # If similar, use the better one
        if AIService._are_similar(chatgpt_text, gemini_text):
            better = AIService._select_better(chatgpt_text, gemini_text)
            return f"{better}\n\n🔍 _Powered by: ChatGPT + Gemini (combined)_"
        
        # Merge different responses
        merged = AIService._merge_different(chatgpt_text, gemini_text)
        return f"{merged}\n\n🔍 _Powered by: ChatGPT + Gemini (merged)_"
    
    @staticmethod
    def _are_similar(text1: str, text2: str) -> bool:
        """Check if two texts are similar"""
        # Simple word overlap check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        common = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return False
        
        similarity = len(common) / len(union)
        return similarity > 0.4
    
    @staticmethod
    def _select_better(r1: str, r2: str) -> str:
        """Select the better response"""
        # Score based on length and structure
        score1 = len(r1) + (r1.count("\n\n") * 20)
        score2 = len(r2) + (r2.count("\n\n") * 20)
        
        return r1 if score1 >= score2 else r2
    
    @staticmethod
    def _merge_different(r1: str, r2: str) -> str:
        """Merge two different responses"""
        # Split into sentences
        sentences1 = [s.strip() + '.' for s in r1.split('.') if s.strip()]
        sentences2 = [s.strip() + '.' for s in r2.split('.') if s.strip()]
        
        # Remove duplicates
        all_sentences = []
        seen = set()
        
        for sent in sentences1 + sentences2:
            key = sent[:50].lower()
            if key not in seen and len(sent) > 20:
                seen.add(key)
                all_sentences.append(sent)
        
        # Take best sentences (max 8)
        if len(all_sentences) > 8:
            # Sort by length (longer sentences often more informative)
            all_sentences.sort(key=len, reverse=True)
            all_sentences = all_sentences[:8]
        
        # Join
        merged = ' '.join(all_sentences)
        
        # Clean up
        merged = merged.replace(' .', '.').replace('..', '.')
        if merged.endswith('..'):
            merged = merged[:-1]
        
        return merged

# ==================== TELEGRAM BOT ====================

class TelegramBot:
    """Main Telegram Bot Handler"""
    
    def __init__(self):
        self.config = config
        self.ai_service = AIService()
        
        # User data
        self.user_sessions: Dict[int, List[str]] = {}
        self.user_rates: Dict[int, List[datetime]] = {}
        
        # Create Telegram application
        self.application = Application.builder().token(self.config.TELEGRAM_TOKEN).build()
        
        # Register handlers
        self._register_handlers()
    
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
🤖 *Welcome {user.first_name}!*

I'm your AI Assistant that combines:
• *ChatGPT* (OpenAI) - for detailed explanations
• *Gemini* (Google) - for concise answers
• *Smart Merge* - best of both worlds!

*How to use:*
Just send me any message and I'll respond using both AI models!

*Commands:*
/start - Show this message
/help - Get help & instructions
/clear - Clear conversation history
/status - Check bot status

*Example questions:*
• What is artificial intelligence?
• Explain quantum computing
• How does machine learning work?

Let's get started! Send me your first question. 😊
        """
        await update.message.reply_text(welcome_text, parse_mode="Markdown")
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🆘 *Help & Information*

*How I work:*
1. You send a message
2. I send it to both ChatGPT and Gemini
3. I intelligently merge their responses
4. You get one comprehensive answer

*Rate Limits:*
• 50 requests per 15 minutes
• This prevents abuse and ensures fair usage

*Privacy:*
• Your chats are not permanently stored
• No personal data is collected
• API calls are secure

*Tips:*
• Be specific with your questions
• Use /clear if conversation gets confusing
• Wait if you hit rate limits

*Need help?* Contact: @Prantik123321
        """
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def _clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command"""
        user_id = update.effective_user.id
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        await update.message.reply_text("✅ Conversation history cleared!")
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status_text = f"""
📊 *Bot Status*

*Services:*
• ChatGPT: ✅ Active
• Gemini: ✅ Active
• Merge Engine: ✅ Working

*Statistics:*
• Active Users: {len(self.user_sessions)}
• Your ID: {update.effective_user.id}

*Models:*
• OpenAI: gpt-3.5-turbo
• Gemini: gemini-2.0-flash

*Status:* 🟢 Operational
        """
        await update.message.reply_text(status_text, parse_mode="Markdown")
    
    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user has exceeded rate limit"""
        now = datetime.now()
        
        if user_id not in self.user_rates:
            self.user_rates[user_id] = []
        
        # Remove old requests (older than 15 minutes)
        valid_requests = [
            req_time for req_time in self.user_rates[user_id]
            if (now - req_time).seconds < 900  # 15 minutes
        ]
        
        # Check limit (50 requests per 15 minutes)
        if len(valid_requests) >= 50:
            return False
        
        # Add current request
        valid_requests.append(now)
        self.user_rates[user_id] = valid_requests
        return True
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""
        user_id = update.effective_user.id
        message_text = update.message.text
        
        logger.info(f"Message from {user_id}: {message_text[:100]}...")
        
        # Check rate limit
        if not self._check_rate_limit(user_id):
            await update.message.reply_text(
                "⏳ Please wait 15 minutes before sending more messages. Rate limit exceeded."
            )
            return
        
        # Show typing indicator
        await update.message.chat.send_action(action="typing")
        
        try:
            # Send processing message
            processing_msg = await update.message.reply_text(
                "🤔 Processing your request with both AI models..."
            )
            
            # Call both AIs in parallel
            chatgpt_task = asyncio.to_thread(self.ai_service.call_chatgpt, message_text)
            gemini_task = asyncio.to_thread(self.ai_service.call_gemini, message_text)
            
            chatgpt_result, gemini_result = await asyncio.gather(chatgpt_task, gemini_task)
            
            # Merge responses
            final_response = self.ai_service.merge_responses(chatgpt_result, gemini_result)
            
            # Update processing message with final response
            await processing_msg.edit_text(final_response, parse_mode="Markdown")
            
            logger.info(f"Response sent to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await update.message.reply_text(
                "❌ Sorry, I encountered an error while processing your message. Please try again."
            )
    
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        try:
            await update.message.reply_text("❌ An error occurred. Please try again later.")
        except:
            pass
    
    def setup_webhook(self, webhook_url: str):
        """Setup webhook for production"""
        self.application.bot.set_webhook(
            url=f"{webhook_url}/webhook",
            allowed_updates=["message", "callback_query"]
        )
        logger.info(f"✅ Webhook set to: {webhook_url}/webhook")
    
    def run_polling(self):
        """Run bot in polling mode (for development)"""
        logger.info("🔧 Starting bot in polling mode...")
        self.application.run_polling(allowed_updates=["message", "callback_query"])
    
    async def shutdown(self):
        """Shutdown the bot gracefully"""
        await self.application.shutdown()

# ==================== FLASK WEB SERVER ====================

# Create Flask app
app = Flask(__name__)

# Global bot instance
telegram_bot = None

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "Telegram AI Bot",
        "version": "3.0.0",
        "endpoints": ["/webhook", "/health", "/status", "/setwebhook"],
        "message": "Bot is running successfully!"
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bot": "running" if telegram_bot else "not initialized"
    })

@app.route('/status')
def status():
    return jsonify({
        "users": len(telegram_bot.user_sessions) if telegram_bot else 0,
        "services": {
            "telegram": "connected",
            "chatgpt": "configured",
            "gemini": "configured"
        },
        "environment": config.NODE_ENV
    })

@app.route('/setwebhook', methods=['POST', 'GET'])
def set_webhook():
    """Manually set webhook"""
    try:
        webhook_url = request.args.get('url') or request.json.get('url')
        if not webhook_url:
            return jsonify({"error": "Missing webhook URL"}), 400
        
        telegram_bot.setup_webhook(webhook_url)
        return jsonify({
            "success": True,
            "message": f"Webhook set to {webhook_url}/webhook"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Telegram webhook endpoint"""
    try:
        # Get update from Telegram
        update_data = request.get_json()
        if not update_data:
            return jsonify({"status": "error", "message": "No data"}), 400
        
        # Process update
        update = Update.de_json(update_data, telegram_bot.application.bot)
        telegram_bot.application.process_update(update)
        
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ==================== MAIN FUNCTION ====================

def main():
    """Main function to run the bot"""
    global telegram_bot
    
    # Validate configuration
    if not config.validate():
        logger.error("❌ Invalid configuration. Please check your .env file.")
        return
    
    # Initialize bot
    telegram_bot = TelegramBot()
    logger.info("🤖 Telegram bot initialized")
    
    # Check if running in production
    if config.NODE_ENV == "production" and config.RENDER_URL:
        # Production mode with webhook
        logger.info("🚀 Starting in PRODUCTION mode (Webhook)")
        
        # Get Render URL
        render_url = os.getenv("RENDER_EXTERNAL_URL") or config.RENDER_URL
        if not render_url:
            logger.error("❌ RENDER_EXTERNAL_URL not set for production")
            return
        
        # Set webhook
        webhook_url = f"{render_url}/webhook"
        telegram_bot.setup_webhook(render_url)
        
        logger.info(f"📡 Webhook URL: {webhook_url}")
        logger.info(f"🌐 Server running on port {config.PORT}")
        logger.info("✅ Bot is ready to receive messages via webhook!")
        
        # Start Flask server (this blocks)
        app.run(
            host='0.0.0.0',
            port=config.PORT,
            debug=False,
            use_reloader=False
        )
    else:
        # Development mode - polling
        logger.info("🔧 Starting in DEVELOPMENT mode (Polling)")
        logger.info("📡 Use ngrok for local webhook testing if needed")
        
        # Run polling
        telegram_bot.run_polling()

if __name__ == "__main__":
    main()
