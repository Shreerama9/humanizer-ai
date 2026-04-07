"use client";

import { useState } from "react";
import { Sparkles, Wand2, BarChart2, Loader2 } from "lucide-react";
import { clsx } from "clsx";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Tab = "generate" | "humanize" | "evaluate";

interface EvalResult {
  composite_score: number;
  recommendation: string;
  seo_score?: { score: number };
  authenticity_score?: { score: number };
  readability_score?: { flesch_reading_ease: number };
}

export default function HomePage() {
  const [tab, setTab] = useState<Tab>("generate");

  // Generate state
  const [keyword, setKeyword] = useState("");
  const [niche, setNiche] = useState("");
  const [format, setFormat] = useState("article");
  const [wordCount, setWordCount] = useState(1200);
  const [tone, setTone] = useState("conversational");

  // Humanize state
  const [inputText, setInputText] = useState("");
  const [intensity, setIntensity] = useState(0.8);

  // Evaluate state
  const [evalText, setEvalText] = useState("");
  const [evalKeyword, setEvalKeyword] = useState("");

  // Results
  const [outputText, setOutputText] = useState("");
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function post(path: string, body: object) {
    const r = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async function handleGenerate() {
    setLoading(true); setError(null); setOutputText("");
    try {
      const data = await post("/generate", {
        keyword, niche, content_format: format,
        target_word_count: wordCount, tone, stream: false,
      });
      setOutputText(data.content);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error");
    } finally { setLoading(false); }
  }

  async function handleHumanize() {
    setLoading(true); setError(null); setOutputText("");
    try {
      const data = await post("/humanize", {
        text: inputText, primary_keyword: keyword, intensity,
      });
      setOutputText(data.humanized_text);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error");
    } finally { setLoading(false); }
  }

  async function handleEvaluate() {
    setLoading(true); setError(null); setEvalResult(null);
    try {
      const data = await post("/evaluate", {
        text: evalText, primary_keyword: evalKeyword,
      });
      setEvalResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error");
    } finally { setLoading(false); }
  }

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "generate", label: "Generate", icon: <Sparkles className="w-4 h-4" /> },
    { id: "humanize", label: "Humanize", icon: <Wand2 className="w-4 h-4" /> },
    { id: "evaluate", label: "Evaluate", icon: <BarChart2 className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">SEO Content Studio</h1>
        <p className="text-sm text-slate-500 mt-1">
          Generate, humanize, and score SEO content with a fine-tuned Llama-3 8B model.
        </p>
      </div>

      {/* Tab Bar */}
      <div className="flex gap-1 bg-slate-100 rounded-lg p-1 w-fit">
        {tabs.map(({ id, label, icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={clsx(
              "flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-colors",
              tab === id ? "bg-white shadow-sm text-slate-900" : "text-slate-500 hover:text-slate-700"
            )}
          >
            {icon} {label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Input */}
        <div className="card space-y-4">
          {tab === "generate" && (
            <>
              <div>
                <label className="text-sm font-medium text-slate-700 block mb-1">Primary Keyword *</label>
                <input className="input" placeholder="e.g. best project management tools" value={keyword} onChange={e => setKeyword(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-700 block mb-1">Niche</label>
                <input className="input" placeholder="e.g. SaaS tools" value={niche} onChange={e => setNiche(e.target.value)} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-sm font-medium text-slate-700 block mb-1">Format</label>
                  <select className="input" value={format} onChange={e => setFormat(e.target.value)}>
                    <option value="article">Article</option>
                    <option value="listicle">Listicle</option>
                    <option value="faq">FAQ</option>
                    <option value="comparison">Comparison</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium text-slate-700 block mb-1">Tone</label>
                  <select className="input" value={tone} onChange={e => setTone(e.target.value)}>
                    <option value="conversational">Conversational</option>
                    <option value="professional">Professional</option>
                    <option value="authoritative">Authoritative</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="text-sm font-medium text-slate-700 block mb-1">Target Word Count: {wordCount}</label>
                <input type="range" min={500} max={3000} step={100} value={wordCount} onChange={e => setWordCount(Number(e.target.value))} className="w-full" />
              </div>
              <button onClick={handleGenerate} disabled={loading || !keyword.trim()} className="btn-primary w-full justify-center">
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                {loading ? "Generating..." : "Generate Content"}
              </button>
            </>
          )}

          {tab === "humanize" && (
            <>
              <div>
                <label className="text-sm font-medium text-slate-700 block mb-1">AI-generated text to humanize *</label>
                <textarea className="input h-48 resize-none" placeholder="Paste AI-generated content here..." value={inputText} onChange={e => setInputText(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-700 block mb-1">Primary Keyword</label>
                <input className="input" placeholder="e.g. project management tools" value={keyword} onChange={e => setKeyword(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-700 block mb-1">Humanization Intensity: {intensity.toFixed(1)}</label>
                <input type="range" min={0.1} max={1.0} step={0.1} value={intensity} onChange={e => setIntensity(Number(e.target.value))} className="w-full" />
                <div className="flex justify-between text-xs text-slate-400 mt-1"><span>Subtle</span><span>Full rewrite</span></div>
              </div>
              <button onClick={handleHumanize} disabled={loading || !inputText.trim()} className="btn-primary w-full justify-center">
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                {loading ? "Humanizing..." : "Humanize Text"}
              </button>
            </>
          )}

          {tab === "evaluate" && (
            <>
              <div>
                <label className="text-sm font-medium text-slate-700 block mb-1">Content to evaluate *</label>
                <textarea className="input h-48 resize-none" placeholder="Paste content here..." value={evalText} onChange={e => setEvalText(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-700 block mb-1">Primary Keyword *</label>
                <input className="input" placeholder="e.g. project management tools" value={evalKeyword} onChange={e => setEvalKeyword(e.target.value)} />
              </div>
              <button onClick={handleEvaluate} disabled={loading || !evalText.trim() || !evalKeyword.trim()} className="btn-primary w-full justify-center">
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <BarChart2 className="w-4 h-4" />}
                {loading ? "Evaluating..." : "Evaluate Content"}
              </button>
            </>
          )}

          {error && <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">{error}</div>}
        </div>

        {/* Right: Output */}
        <div className="card">
          {tab === "evaluate" && evalResult ? (
            <div className="space-y-5">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-slate-900">Evaluation Results</h3>
                <span className={clsx(
                  "badge",
                  evalResult.recommendation === "PUBLISH" ? "bg-green-100 text-green-700" :
                  evalResult.recommendation === "REVISE" ? "bg-yellow-100 text-yellow-700" : "bg-red-100 text-red-700"
                )}>
                  {evalResult.recommendation}
                </span>
              </div>
              <div className="text-center py-4">
                <div className="text-5xl font-bold text-violet-600">{evalResult.composite_score?.toFixed(1)}</div>
                <div className="text-sm text-slate-400 mt-1">Composite Score / 100</div>
              </div>
              {[
                { label: "SEO Score", value: evalResult.seo_score?.score },
                { label: "Authenticity", value: evalResult.authenticity_score?.score },
                { label: "Readability (Flesch)", value: evalResult.readability_score?.flesch_reading_ease },
              ].map(({ label, value }) => value !== undefined && (
                <div key={label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-600">{label}</span>
                    <span className="font-medium">{value?.toFixed(1)}</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-violet-500 rounded-full" style={{ width: `${Math.min(value || 0, 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          ) : outputText ? (
            <div className="space-y-3 h-full">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-slate-900">Output</h3>
                <button onClick={() => navigator.clipboard.writeText(outputText)} className="text-xs text-slate-400 hover:text-slate-600">Copy</button>
              </div>
              <pre className="text-sm text-slate-700 whitespace-pre-wrap font-sans leading-relaxed max-h-96 overflow-y-auto">{outputText}</pre>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center py-16">
              <div>
                <Sparkles className="w-10 h-10 text-slate-200 mx-auto mb-3" />
                <p className="text-sm text-slate-400">Output will appear here.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
