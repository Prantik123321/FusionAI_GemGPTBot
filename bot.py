#!/usr/bin/env python3
"""
Telegram AI Chatbot - ChatGPT + Gemini
Author: Prantik
Version: 1.0.0
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dataclasses import dataclass
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import libraries
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application, 
        CommandHandler, 
        MessageHandler, 
        filters, 
        ContextTypes,
        CallbackQueryHandler
    )
    import openai
    import google.generativeai as genai
    import requests
except ImportError as e:
    print(f"Error: Missing library - {e}")
    print("Please install requirements: pip install -r requirements.txt")
    exit(1)

# ==================== CONFIGURATION ====================
@dataclass
class Config:
    """Configuration class for API keys and settings"""
    
    # Load from environment
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    PORT: int = int(os.getenv("PORT", "10000"))
    NODE_ENV: str = os.getenv("NODE_ENV", "production")
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 50  # requests per user
    RATE_LIMIT_WINDOW: int = 900   # 15 minutes in seconds
    
    # AI Models
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    def validate(self) -> bool:
        """Validate all configuration values"""
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

# Initialize config
config = Config()

# ==================== AI SERVICES ====================

class ChatGPTService:
    """ChatGPT API Service"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.OPENAI_MODEL
        
    async def generate_response(self, prompt: str, context: str = "") -> Dict:
        """Generate response from ChatGPT"""
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise, and accurate responses."}
            ]
            
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
                
            messages.append({"role": "user", "content": prompt})
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=600,
                temperature=0.7
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content.strip(),
                "model": self.model,
                "tokens": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"ChatGPT error: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": True
            }

class GeminiService:
    """Google Gemini API Service"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        
    async def generate_response(self, prompt: str, context: str = "") -> Dict:
        """Generate response from Gemini"""
        try:
            # Prepare prompt with context
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            # Call Gemini API
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 600,
                }
            )
            
            return {
                "success": True,
                "content": response.text.strip(),
                "model": config.GEMINI_MODEL
            }
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": True
            }

class ResponseMerger:
    """Merge responses from both AI models"""
    
    @staticmethod
    def merge_responses(chatgpt_result: Dict, gemini_result: Dict) -> Dict:
        """Intelligently merge two AI responses"""
        
        # Handle failures
        if not chatgpt_result["success"] and not gemini_result["success"]:
            return {
                "content": "I apologize, but both AI services are currently unavailable. Please try again later.",
                "sources": [],
                "note": "Service outage"
            }
            
        if not chatgpt_result["success"]:
            return {
                "content": gemini_result["content"],
                "sources": ["Gemini"],
                "note": "Using Gemini only"
            }
            
        if not gemini_result["success"]:
            return {
                "content": chatgpt_result["content"],
                "sources": ["ChatGPT"],
                "note": "Using ChatGPT only"
            }
            
        # Both succeeded - merge intelligently
        chatgpt_text = chatgpt_result["content"]
        gemini_text = gemini_result["content"]
        
        # If responses are similar, use the better one
        if ResponseMerger._are_similar(chatgpt_text, gemini_text):
            better_response = ResponseMerger._select_better(chatgpt_text, gemini_text)
            return {
                "content": better_response,
                "sources": ["ChatGPT", "Gemini"],
                "note": "Combined insights"
            }
            
        # Merge different responses
        merged = ResponseMerger._merge_different(chatgpt_text, gemini_text)
        return {
            "content": merged,
            "sources": ["ChatGPT", "Gemini"],
            "note": "Merged from both AIs"
        }
    
    @staticmethod
    def _are_similar(text1: str, text2: str) -> bool:
        """Check if two responses are similar"""
        t1 = text1.lower().replace("\n", " ").strip()
        t2 = text2.lower().replace("\n", " ").strip()
        
        # Simple word overlap check
        words1 = set(t1.split())
        words2 = set(t2.split())
        
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
        score1 = len(r1) + (r1.count("\n\n") * 10)
        score2 = len(r2) + (r2.count("\n\n") * 10)
        
        return r1 if score1 >= score2 else r2
    
    @staticmethod
    def _merge_different(r1: str, r2: str) -> str:
        """Merge two different responses"""
        # Split into sentences
        def split_sentences(text):
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
        
        sentences1 = split_sentences(r1)
        sentences2 = split_sentences(r2)
        
        # Remove duplicates
        all_sentences = []
        seen = set()
        
        for sent in sentences1 + sentences2:
            key = sent[:50].lower()
            if key not in seen and len(sent) > 25:
                seen.add(key)
                all_sentences.append(sent)
        
        # Take top sentences (limit to 10)
        selected = all_sentences[:10]
        
        # Join into coherent response
        merged = " ".join(selected)
        
        # Ensure it ends with punctuation
        if merged and merged[-1] not in ".!?":
            merged += "."
            
        return merged

# ==================== BOT HANDLER ====================

class TelegramBotHandler:
    """Main Telegram bot handler"""
    
    def __init__(self):
        self.config = config
        self.chatgpt = ChatGPTService(config.OPENAI_API_KEY)
        self.gemini = GeminiService(config.GEMINI_API_KEY)
        self.merger = ResponseMerger()
        
        # User data storage
        self.user_sessions: Dict[int, Dict] = {}  # user_id -> session data
        self.user_rate_limits: Dict[int, List[datetime]] = {}  # user_id -> request times
        
        # Initialize bot
        self.application = Application.builder().token(config.TELEGRAM_TOKEN).build()
        
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
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        
        # Error handler
        self.application.add_error_handler(self._error_handler)
        
    # ========== COMMAND HANDLERS ==========
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        welcome_text = f"""
🤖 *Welcome {user.first_name}!*

I'm your AI Assistant bot that combines:
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
• Explain quantum computing
• What is machine learning?
• How does AI work?

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
• {config.RATE_LIMIT_REQUESTS} requests per 15 minutes
• This prevents abuse and ensures fair usage

*Privacy:*
• Your chats are not permanently stored
• No personal data is collected
• API calls are secure

*Tips:*
• Be specific with your questions
• Use /clear if conversation gets confusing
• Wait if you hit rate limits

*Need help?*
Contact: @Prantik123321
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
        user_id = update.effective_user.id
        
        status_text = f"""
📊 *Bot Status*

*Services:*
• ChatGPT: ✅ Active
• Gemini: ✅ Active
• Merge Engine: ✅ Working

*Statistics:*
• Active Users: {len(self.user_sessions)}
• Your Requests: {len(self.user_rate_limits.get(user_id, []))}
• Rate Limit: {config.RATE_LIMIT_REQUESTS}/15min

*Models:*
• OpenAI: {config.OPENAI_MODEL}
• Gemini: {config.GEMINI_MODEL}

*Status:* 🟢 Operational
        """
        
        await update.message.reply_text(status_text, parse_mode="Markdown")
        
    # ========== MESSAGE HANDLER ==========
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""
        user_id = update.effective_user.id
        message_text = update.message.text
        
        # Check rate limit
        if not self._check_rate_limit(user_id):
            await update.message.reply_text(
                "⏳ Please wait 15 minutes before sending more messages. Rate limit exceeded."
            )
            return
            
        # Show typing indicator
        await update.message.chat.send_action(action="typing")
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🤔 Processing your request with both AI models..."
        )
        
        try:
            # Get user context
            user_context = self._get_user_context(user_id)
            
            # Call both AI APIs in parallel
            chatgpt_task = self.chatgpt.generate_response(message_text, user_context)
            gemini_task = self.gemini.generate_response(message_text, user_context)
            
            chatgpt_result, gemini_result = await asyncio.gather(
                chatgpt_task, 
                gemini_result,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(chatgpt_result, Exception):
                chatgpt_result = {"success": False, "error": str(chatgpt_result)}
            if isinstance(gemini_result, Exception):
                gemini_result = {"success": False, "error": str(gemini_result)}
            
            # Merge responses
            merged_result = self.merger.merge_responses(chatgpt_result, gemini_result)
            
            # Update user context
            self._update_user_context(user_id, message_text, merged_result["content"])
            
            # Format final message
            final_message = merged_result["content"]
            
            # Add footer with sources
            if merged_result["sources"]:
                final_message += f"\n\n---\n"
                final_message += f"_Powered by: {' + '.join(merged_result['sources'])}_"
                if merged_result.get("note"):
                    final_message += f"\n_({merged_result['note']})_"
            
            # Edit processing message with final response
            await processing_msg.edit_text(final_message, parse_mode="Markdown")
            
            logger.info(f"Response sent to user {user_id}. Sources: {merged_result['sources']}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await processing_msg.edit_text(
                "❌ Sorry, I encountered an error. Please try again."
            )
            
    # ========== UTILITY METHODS ==========
    
    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user has exceeded rate limit"""
        now = datetime.now()
        
        if user_id not in self.user_rate_limits:
            self.user_rate_limits[user_id] = []
            
        # Remove old requests
        valid_requests = [
            req_time for req_time in self.user_rate_limits[user_id]
            if (now - req_time).seconds < config.RATE_LIMIT_WINDOW
        ]
        
        # Check if limit exceeded
        if len(valid_requests) >= config.RATE_LIMIT_REQUESTS:
            return False
            
        # Add current request
        valid_requests.append(now)
        self.user_rate_limits[user_id] = valid_requests
        
        return True
        
    def _get_user_context(self, user_id: int) -> str:
        """Get user's conversation context"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "messages": [],
                "created_at": datetime.now()
            }
            
        session = self.user_sessions[user_id]
        # Return last 3 interactions
        return "\n".join(session["messages"][-6:])  # Last 3 Q&A pairs
        
    def _update_user_context(self, user_id: int, user_msg: str, ai_response: str):
        """Update user's conversation context"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "messages": [],
                "created_at": datetime.now()
            }
            
        session = self.user_sessions[user_id]
        session["messages"].append(f"User: {user_msg}")
        session["messages"].append(f"AI: {ai_response}")
        
        # Keep only last 6 messages (3 interactions)
        if len(session["messages"]) > 6:
            session["messages"] = session["messages"][-6:]
            
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in the bot"""
        logger.error(f"Update {update} caused error {context.error}")
        
        # Try to notify user
        try:
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "❌ An error occurred. Please try again later."
                )
        except:
            pass
            
    # ========== PUBLIC METHODS ==========
    
    async def start_polling(self):
        """Start the bot in polling mode (for development)"""
        logger.info("Starting bot in polling mode...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
    async def stop_polling(self):
        """Stop the bot"""
        await self.application.stop()
        
    def run_polling(self):
        """Run the bot with polling (blocking)"""
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    def setup_webhook(self, webhook_url: str):
        """Setup webhook for production"""
        self.application.bot.set_webhook(
            url=f"{webhook_url}/webhook",
            allowed_updates=Update.ALL_TYPES
        )

# ==================== WEB SERVER (for Render) ====================

from flask import Flask, request, jsonify

def create_web_server(bot_handler: TelegramBotHandler):
    """Create Flask web server for webhook"""
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return jsonify({
            "status": "online",
            "service": "Telegram AI Bot",
            "version": "1.0.0",
            "endpoints": ["/webhook", "/health", "/status"]
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/status')
    def status():
        return jsonify({
            "users": len(bot_handler.user_sessions),
            "rate_limits": len(bot_handler.user_rate_limits),
            "services": {
                "telegram": "connected",
                "chatgpt": "configured",
                "gemini": "configured"
            }
        })
    
    @app.route('/webhook', methods=['POST'])
    def webhook():
        """Telegram webhook endpoint"""
        update = Update.de_json(request.get_json(), bot_handler.application.bot)
        bot_handler.application.process_update(update)
        return jsonify({"status": "ok"})
    
    return app

# ==================== MAIN ENTRY POINT ====================

async def main():
    """Main function to run the bot"""
    
    # Validate configuration
    if not config.validate():
        logger.error("Invalid configuration. Please check your .env file.")
        return
    
    # Initialize bot handler
    bot_handler = TelegramBotHandler()
    
    # Check if running in production (Render)
    if config.NODE_ENV == "production":
        logger.info("🚀 Starting in production mode (Webhook)")
        
        # Create web server
        app = create_web_server(bot_handler)
        
        # Note: Webhook needs to be set manually
        # Command: curl -X POST "https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={RENDER_URL}/webhook"
        
        # Start Flask server
        import threading
        
        def run_flask():
            app.run(host='0.0.0.0', port=config.PORT)
            
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.start()
        
        logger.info(f"✅ Web server started on port {config.PORT}")
        logger.info(f"📡 Webhook URL: https://your-app.onrender.com/webhook")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            
    else:
        # Development mode - use polling
        logger.info("🔧 Starting in development mode (Polling)")
        bot_handler.run_polling()

if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())
