'use client';

import React from 'react';
import { Language } from '../lib/i18n';
import { Globe } from 'lucide-react';

interface LanguageSelectorProps {
  lang: Language;
  setLang: (lang: Language) => void;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({ lang, setLang }) => {
  return (
    <div className="flex items-center space-x-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-lg border border-slate-200 dark:border-slate-700">
      <Globe className="w-4 h-4 ml-1.5 text-slate-500 dark:text-slate-400" />
      <button
        onClick={() => setLang('es')}
        className={`px-2.5 py-1 text-xs font-semibold rounded ${
          lang === 'es' ? 'bg-teal-600 text-white shadow-sm' : 'text-slate-600 dark:text-slate-300 hover:text-slate-900'
        }`}
      >
        ES
      </button>
      <button
        onClick={() => setLang('en')}
        className={`px-2.5 py-1 text-xs font-semibold rounded ${
          lang === 'en' ? 'bg-teal-600 text-white shadow-sm' : 'text-slate-600 dark:text-slate-300 hover:text-slate-900'
        }`}
      >
        EN
      </button>
    </div>
  );
};
