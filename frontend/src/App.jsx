import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ShieldAlert, CheckCircle2, BookOpen, User, Stethoscope, Send, Activity, AlertTriangle } from 'lucide-react';

const API_BASE = "http://localhost:8000";

export default function App() {
  const [drugs, setDrugs] = useState([]);
  const [selectedDrug, setSelectedDrug] = useState('metformin');
  const [audience, setAudience] = useState('patient');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/drugs`)
      .then(res => setDrugs(res.data.drugs))
      .catch(err => console.error("Error loading drugs:", err));
  }, []);

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResponse(null);

    try {
      const res = await axios.post(`${API_BASE}/ask`, {
        drug: selectedDrug,
        query: query,
        audience: audience
      });
      setResponse(res.data);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
      {/* Top Header */}
      <header className="bg-teal-700 text-white shadow-md py-4 px-6 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="h-7 w-7 text-teal-200" />
          <h1 className="text-xl font-bold tracking-tight">Ritam AI</h1>
          <span className="bg-teal-800 text-teal-100 text-xs px-3 py-1 rounded-full font-mono uppercase tracking-wider">
            Verified FDA Medical Engine
          </span>
        </div>
        <div className="flex items-center space-x-2 text-xs bg-teal-800/60 px-3 py-1.5 rounded-full text-teal-100">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>Guardrails Active</span>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Control Panel */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 space-y-6 h-fit">
          <div>
            <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
              Select Drug Context
            </label>
            <select 
              value={selectedDrug} 
              onChange={(e) => setSelectedDrug(e.target.value)}
              className="w-full p-3 bg-slate-50 border border-slate-300 rounded-lg focus:ring-2 focus:ring-teal-500 font-medium text-slate-700"
            >
              {drugs.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">
              Target Audience Mode
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setAudience('patient')}
                className={`flex items-center justify-center space-x-2 py-2.5 px-3 rounded-lg border text-sm font-medium transition-all ${
                  audience === 'patient' 
                    ? 'bg-teal-50 border-teal-600 text-teal-700 font-semibold' 
                    : 'border-slate-200 hover:bg-slate-50 text-slate-600'
                }`}
              >
                <User className="w-4 h-4" />
                <span>Patient</span>
              </button>
              <button
                type="button"
                onClick={() => setAudience('clinician')}
                className={`flex items-center justify-center space-x-2 py-2.5 px-3 rounded-lg border text-sm font-medium transition-all ${
                  audience === 'clinician' 
                    ? 'bg-teal-50 border-teal-600 text-teal-700 font-semibold' 
                    : 'border-slate-200 hover:bg-slate-50 text-slate-600'
                }`}
              >
                <Stethoscope className="w-4 h-4" />
                <span>Clinician</span>
              </button>
            </div>
          </div>

          <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-600 space-y-2">
            <div className="font-semibold text-slate-700 flex items-center gap-1.5">
              <BookOpen className="w-4 h-4 text-teal-600" /> Grounded RAG Pipeline
            </div>
            <p>Every response is generated directly from official FDA drug labels with full source citation and quote verification.</p>
          </div>
        </div>

        {/* Workspace Display */}
        <div className="lg:col-span-2 space-y-6 flex flex-col">
          
          {/* Query Bar */}
          <form onSubmit={handleAsk} className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 flex gap-3">
            <input 
              type="text"
              placeholder={`Ask about ${selectedDrug}... (e.g. "What are common side effects?")`}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 border-none focus:outline-none text-slate-700 placeholder-slate-400 text-base px-2"
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-teal-600 hover:bg-teal-700 text-white font-medium px-5 py-2.5 rounded-lg flex items-center space-x-2 transition-all disabled:opacity-50"
            >
              <span>{loading ? "Analyzing..." : "Ask"}</span>
              <Send className="w-4 h-4" />
            </button>
          </form>

          {/* Results Area */}
          {response && (
            <div className="space-y-4">
              
              {/* Emergency Intercept */}
              {response.type === "emergency" ? (
                <div className="bg-red-50 border-2 border-red-500 rounded-xl p-5 text-red-900 space-y-3">
                  <div className="flex items-center space-x-3">
                    <ShieldAlert className="w-7 h-7 text-red-600" />
                    <h3 className="text-lg font-bold">Emergency Detected</h3>
                  </div>
                  <p className="font-medium">{response.message}</p>
                  <div className="bg-white p-4 rounded-lg border border-red-200 space-y-1 text-sm">
                    <p><strong>Emergency Contact:</strong> {response.emergency_info?.primary?.tel} ({response.emergency_info?.primary?.label})</p>
                    <p><strong>Poison Control:</strong> {response.emergency_info?.poison?.tel} ({response.emergency_info?.poison?.label})</p>
                  </div>
                </div>
              ) : (
                /* Verified AI Output */
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 space-y-5">
                  <div>
                    <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Verified Response</h3>
                    <p className="text-slate-800 text-base leading-relaxed whitespace-pre-wrap">{response.answer}</p>
                  </div>

                  {/* Audit Badges */}
                  {response.verification && (
                    <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-slate-100 text-xs">
                      <div className="flex items-center space-x-1.5 bg-slate-100 text-slate-700 px-3 py-1.5 rounded-full font-medium">
                        <CheckCircle2 className="w-4 h-4 text-teal-600" />
                        <span>Quotes Valid: <strong>{response.verification.quotes_valid ? "PASS" : "FAIL"}</strong></span>
                      </div>
                      <div className="flex items-center space-x-1.5 bg-slate-100 text-slate-700 px-3 py-1.5 rounded-full font-medium">
                        <CheckCircle2 className="w-4 h-4 text-teal-600" />
                        <span>Numbers Valid: <strong>{response.verification.numbers_valid ? "PASS" : "FAIL"}</strong></span>
                      </div>
                      {response.readability && (
                        <div className="flex items-center space-x-1.5 bg-slate-100 text-slate-700 px-3 py-1.5 rounded-full font-medium">
                          <span>Grade Level: <strong>{response.readability.flesch_kincaid_grade}</strong></span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* FDA Source Citations */}
                  {response.sources && response.sources.length > 0 && (
                    <div className="pt-4 border-t border-slate-100">
                      <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">FDA Source References</h4>
                      <div className="space-y-2">
                        {response.sources.map((src, i) => (
                          <div key={i} className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-xs text-slate-600">
                            <span className="font-semibold text-slate-800">Section: {src.section} (Page {src.page})</span>
                            <p className="mt-1 italic">"{src.text}"</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}