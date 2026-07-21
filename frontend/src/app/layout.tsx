import './globals.css';
import React from 'react';

export const metadata = {
  title: 'DermAI & Agricultural AI Platform',
  description: 'Full-stack platform with 3 Classic + 2 Hybrid models, 5-Fold CV, Robust Statistical Validation & Reports',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body className="bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 min-h-screen transition-colors duration-200">
        {children}
      </body>
    </html>
  );
}
