'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Language, translations } from '../lib/i18n';
import { MessageSquare, Mic, Send, Trash2, Bot, User, Volume2 } from 'lucide-react';

interface VoiceChatbotProps {
  lang: Language;
}

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

export function VoiceChatbot({ lang }: VoiceChatbotProps) {
  const t = translations[lang];
  const [messages, setMessages] = useState<Message[]>([
    { sender: 'bot', text: t.welcome_msg }
  ]);
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = input;
    setMessages((prev) => [...prev, { sender: 'user', text: userMsg }]);
    setInput('');

    try {
      const res = await fetch('http://localhost:8000/api/chatbot/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg, language: lang })
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { sender: 'bot', text: data.reply }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: 'Error al conectar con el servidor del chatbot.' }
      ]);
    }
  };

  const startVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Tu navegador no soporta entrada de voz por SpeechRecognition.');
      return;
    }
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = lang === 'es' ? 'es-ES' : 'en-US';

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
    };

    recognition.start();
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col h-[600px] w-full">
      {/* Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between bg-slate-50 dark:bg-slate-900 rounded-t-xl">
        <div className="flex items-center space-x-2">
          <MessageSquare className="w-5 h-5 text-teal-600 dark:text-teal-400" />
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">{t.chatbot_title}</h3>
        </div>
        <button
          onClick={() => setMessages([{ sender: 'bot', text: t.welcome_msg }])}
          className="p-1.5 text-slate-500 hover:text-red-600 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg transition-colors"
          title={t.clear_chat}
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 p-4 overflow-y-auto space-y-3">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex items-start space-x-2 ${
              m.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
            }`}
          >
            <div
              className={`p-2 rounded-full ${
                m.sender === 'user'
                  ? 'bg-teal-600 text-white'
                  : 'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200'
              }`}
            >
              {m.sender === 'user' ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
            </div>

            <div
              className={`max-w-[80%] p-3 rounded-xl text-xs leading-relaxed ${
                m.sender === 'user'
                  ? 'bg-teal-600 text-white rounded-tr-none shadow-sm'
                  : 'bg-slate-100 dark:bg-slate-900 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 rounded-tl-none shadow-sm'
              }`}
            >
              {m.text}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input controls */}
      <div className="p-3 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 rounded-b-xl space-y-2">
        <div className="flex items-center space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder={t.chatbot_placeholder}
            className="flex-1 px-3 py-2 text-xs border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500"
          />

          <button
            onClick={startVoiceInput}
            className={`p-2 rounded-lg transition-colors ${
              isListening
                ? 'bg-red-600 text-white animate-pulse'
                : 'bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200'
            }`}
            title={t.speak_btn}
          >
            <Mic className="w-4 h-4" />
          </button>

          <button
            onClick={handleSend}
            className="p-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg transition-colors"
            title={t.send_btn}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
