'use client';

import { useState } from 'react';

export default function Dashboard() {
  const [query, setQuery] = useState('');
  const [report, setReport] = useState('');
  const [loading, setLoading] = useState(false);

  const askBackend = async () => {
    if (!query) return;
    setLoading(true);
    setReport('');
    
    try {
      const res = await fetch('http://127.0.0.1:8000/api/ask-agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          user_query: query, 
          cv_detections: []
        })
      });
      const data = await res.json();
      setReport(data.report);
    } catch (error) {
      setReport("Connection failed. Please ensure the backend server is running on port 8000.");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-cyan-500 selection:text-white">
      <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-[0_0_15px_rgba(6,182,212,0.5)]">
                AI
              </div>
              <span className="font-bold text-xl tracking-tight text-white">Visual Compliance</span>
            </div>
            <div className="flex gap-4">
              <span className="px-3 py-1 text-xs font-medium bg-emerald-500/10 text-emerald-400 rounded-full border border-emerald-500/20 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                System Online
              </span>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          <div className="flex flex-col gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden group hover:border-slate-700 transition-all duration-300">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 to-blue-500"></div>
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                Live Detection Feed
              </h2>
              
              <div className="aspect-video bg-slate-950 rounded-xl border border-slate-800 border-dashed flex flex-col items-center justify-center text-slate-500 group-hover:bg-slate-900/50 transition-colors cursor-pointer">
                <svg className="w-12 h-12 mb-3 text-slate-600 group-hover:text-cyan-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                <p className="font-medium text-slate-300">Click to upload footage</p>
                <p className="text-sm mt-1">MP4, AVI, or JPG</p>
              </div>

              <div className="mt-6 flex justify-between items-center bg-slate-950/50 p-4 rounded-lg border border-slate-800/50">
                <div className="text-center">
                  <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Status</p>
                  <p className="font-semibold text-slate-300">Awaiting Upload</p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Model</p>
                  <p className="font-semibold text-cyan-400">YOLOv8 Custom</p>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden h-full flex flex-col hover:border-slate-700 transition-all duration-300">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-pink-500"></div>
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
                AI Safety Officer (RAG)
              </h2>

              <div className="flex-1 bg-slate-950 rounded-xl border border-slate-800 p-4 mb-4 flex flex-col overflow-y-auto relative min-h-[300px]">
                {report ? (
                  <div className="text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">
                    {report}
                  </div>
                ) : (
                  <div className="text-center absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                    <svg className="w-10 h-10 mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
                    <p>Connected to ChromaDB</p>
                    <p className="text-xs mt-1">Ask a question about safety policies...</p>
                  </div>
                )}
              </div>

              <div className="relative mt-auto">
                <input 
                  type="text" 
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && askBackend()}
                  placeholder="e.g., What is the penalty for not wearing a helmet?" 
                  className="w-full bg-slate-950 border border-slate-700 text-slate-200 rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-all"
                />
                <button 
                  onClick={askBackend}
                  disabled={loading}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white rounded-lg transition-colors">
                  {loading ? (
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin inline-block"></span>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
                  )}
                </button>
              </div>
            </div>
          </div>
          
        </div>
      </main>
    </div>
  );
}
