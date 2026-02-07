#!/usr/bin/env python3
"""
Simple Telegram AI Bot - ChatGPT + Gemini
Python 3.13 compatible version
"""

import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import libraries
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import openai
import google.generativeai as genai
from flask import Flask, request, jsonify

# Configuration
class Config:
    def __init__(self):
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8203704693:AAFmr89RSLcW1HfOR_SXRC9X2bDhhMnT7EY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBhQybaUJ02OJdy2uqbuN_oYgrWReT2aBY")
        self.PORT = int(os.getenv("PORT", "10000"))
        
    def validate(self):
        """Basic validation"""
        required = ['TELEGRAM_TOKEN', 'OPENAI_API_KEY', 'GEMINI_API_KEY']
        for key in required:
            if not getattr(self, key):
                logger.error(f"Missing {key}")
                return False
        logger.info("✅ Configuration validated")
        return True

config = Config()

# Initialize AI services
openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
genai.configure(api_key=config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# User data storage
user_sessions = {}
rate_limits = {}

# Flask app for webhook
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "Telegram AI Bot",
        "version": "2.0.0",
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
        "users": len(user_sessions),
        "services": {
            "telegram": "configured",
            "chatgpt": "configured", 
            "gemini": "configured"
        }
    })

# Telegram Bot Functions
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 Welcome to AI Assistant Bot!\n\n"
        "I combine ChatGPT (OpenAI) and Gemini (Google) for better answers.\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/help - Get help\n"
        "/clear - Clear history\n\n"
        "Just send me any message! 😊"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await update.message.reply_text(
        "🆘 Help Guide\n\n"
        "• I use both ChatGPT and Gemini\n"
        "• Responses are merged intelligently\n"
        "• Rate limit: 50 requests per 15 minutes\n\n"
        "Just type your question and I'll respond!"
    )

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command"""
    user_id = update.effective_user.id
    if user_id in user_sessions:
        del user_sessions[user_id]
    await update.message.reply_text("✅ Conversation cleared!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular messages"""
    user_id = update.effective_user.id
    message = update.message.text
    
    logger.info(f"Message from {user_id}: {message[:50]}...")
    
    # Show typing indicator
    await update.message.chat.send_action(action="typing")
    
    try:
        # Send processing message
        processing_msg = await update.message.reply_text("🤔 Processing with both AIs...")
        
        # Get responses from both AIs
        chatgpt_response = get_chatgpt_response(message)
        gemini_response = get_gemini_response(message)
        
        # Merge responses
        final_response = merge_responses(chatgpt_response, gemini_response)
        
        # Update message with response
        await processing_msg.edit_text(final_response)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ Error processing request. Please try again.")

def get_chatgpt_response(prompt):
    """Get response from ChatGPT"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "source": "ChatGPT"
        }
    except Exception as e:
        logger.error(f"ChatGPT error: {e}")
        return {
            "success": False,
            "content": "",
            "source": "ChatGPT"
        }

def get_gemini_response(prompt):
    """Get response from Gemini"""
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
            "content": response.text,
            "source": "Gemini"
        }
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return {
            "success": False,
            "content": "",
            "source": "Gemini"
        }

def merge_responses(chatgpt_res, gemini_res):
    """Simple response merger"""
    if not chatgpt_res["success"] and not gemini_res["success"]:
        return "Sorry, both AI services are unavailable. Please try again later."
    
    if not chatgpt_res["success"]:
        return f"{gemini_res['content']}\n\n🔍 Powered by: {gemini_res['source']}"
    
    if not gemini_res["success"]:
        return f"{chatgpt_res['content']}\n\n🔍 Powered by: {chatgpt_res['source']}"
    
    # Both succeeded - use the longer response
    chatgpt_text = chatgpt_res["content"]
    gemini_text = gemini_res["content"]
    
    # Simple merge: take first half from one, second half from another
    chatgpt_sentences = chatgpt_text.split('. ')
    gemini_sentences = gemini_text.split('. ')
    
    # Take 3-4 sentences from each
    merged_sentences = []
    if chatgpt_sentences:
        merged_sentences.extend(chatgpt_sentences[:min(3, len(chatgpt_sentences))])
    if gemini_sentences:
        merged_sentences.extend(gemini_sentences[:min(3, len(gemini_sentences))])
    
    merged_text = '. '.join(merged_sentences)
    if merged_text and not merged_text.endswith('.'):
        merged_text += '.'
    
    return f"{merged_text}\n\n🔍 Powered by: ChatGPT + Gemini"

# Webhook endpoint for Telegram
@app.route('/webhook', methods=['POST'])
def webhook():
    """Telegram webhook endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data"}), 400
        
        # Process update
        update = Update.de_json(data, bot)
        application.process_update(update)
        
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def setup_bot():
    """Setup and configure the bot"""
    # Create application
    application = Application.builder().token(config.TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    return application

def main():
    """Main function to run the bot"""
    if not config.validate():
        logger.error("Invalid configuration. Exiting.")
        return
    
    logger.info("🚀 Starting Telegram AI Bot...")
    
    # Get bot application
    global application, bot
    application = setup_bot()
    bot = application.bot
    
    # Get webhook URL from environment
    webhook_url = os.getenv("RENDER_EXTERNAL_URL", "")
    
    if webhook_url:
        # Production mode - use webhook
        logger.info(f"📡 Setting up webhook for: {webhook_url}")
        
        # Set webhook
        webhook_url_full = f"{webhook_url}/webhook"
        bot.set_webhook(url=webhook_url_full)
        
        logger.info(f"✅ Webhook set to: {webhook_url_full}")
        logger.info(f"🌐 Server starting on port {config.PORT}")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=config.PORT, debug=False)
    else:
        # Development mode - use polling
        logger.info("🔧 Starting in development mode (polling)")
        application.run_polling()

if __name__ == "__main__":
    main()
