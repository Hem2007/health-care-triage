import React, { useState, useEffect, FormEvent, useRef } from 'react';
import { AlertCircle, FileUp, Info, Activity, Moon, Sun, AlertTriangle, ShieldAlert, CheckCircle, Skull, UploadCloud, Download } from 'lucide-react';

const API_URL = '/api';

// Types
type Tab = 'triage' | 'batch' | 'report' | 'about';
type FieldSchema = {
  name: string;
  type: 'numeric' | 'categorical' | 'text';
  values?: any[];
  hint?: string;
  range?: [number, number];
};

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('triage');
  const [darkMode, setDarkMode] = useState(false);
  const [apiHealth, setApiHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const [schema, setSchema] = useState<{ core: FieldSchema[], advanced: FieldSchema[] } | null>(null);
  
  // Fetch schema and health on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.json())
      .then(setApiHealth)
      .catch(() => setApiHealth({ status: 'down', model_loaded: false }));
      
    fetch(`${API_URL}/schema`)
      .then(r => r.json())
      .then(setSchema)
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (darkMode) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [darkMode]);

  return (
    <div className="min-h-screen flex flex-col font-sans">
            {/* Header */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2 text-xl font-bold text-slate-800 dark:text-slate-100">
            <Activity className="text-blue-600 dark:text-blue-400" />
            Triage AI
          </div>
          
          <nav className="flex overflow-x-auto w-full sm:w-auto gap-1 bg-slate-100 dark:bg-slate-900 p-1 rounded-lg">
            {[
              { id: 'triage', label: 'Triage a patient', icon: Activity },
              { id: 'batch', label: 'Batch upload', icon: FileUp },
              { id: 'report', label: 'Model report', icon: Info },
              { id: 'about', label: 'How it works', icon: ShieldAlert }
            ].map(tab => (
              <button 
                key={tab.id}
                onClick={() => setActiveTab(tab.id as Tab)}
                className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium whitespace-nowrap transition-colors ${
                  activeTab === tab.id 
                    ? 'bg-white dark:bg-slate-800 text-blue-600 dark:text-blue-400 shadow-sm' 
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                }`}
              >
                <tab.icon size={16} />
                {tab.label}
              </button>
            ))}
          </nav>
          
          <button 
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300"
            aria-label="Toggle dark mode"
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="flex-grow max-w-6xl w-full mx-auto px-4 py-8">
        {apiHealth?.status === 'down' ? (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6 text-center">
            <AlertCircle className="mx-auto text-red-500 mb-4" size={48} />
            <h2 className="text-xl font-bold text-red-700 dark:text-red-400 mb-2">API is unreachable</h2>
            <p className="text-red-600 dark:text-red-300">Ensure the FastAPI backend is running on port 8000.</p>
          </div>
        ) : (
          <>
            {activeTab === 'triage' && schema && <TriageTab schema={schema} />}
            {activeTab === 'batch' && <BatchTab />}
            {activeTab === 'report' && <ReportTab />}
            {activeTab === 'about' && <AboutTab />}
          </>
        )}
      </main>
      
      <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 py-6 mt-auto">
        <div className="max-w-6xl mx-auto px-4 text-center text-sm text-slate-500 dark:text-slate-400">
          MM26ML03 Healthcare Triage Classification &copy; {new Date().getFullYear()}
        </div>
      </footer>
    </div>
  );
}

// ---------------------------
// Triage Tab
// ---------------------------
function TriageTab({ schema }: { schema: { core: FieldSchema[], advanced: FieldSchema[] } }) {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadExample = (type: 'severe' | 'moderate' | 'minor') => {
    if (type === 'severe') {
      setFormData({
        resprate: 0, heartrate: 0, gcs_eye: 1, gcs_verbal: 1, gcs_motor: 1, 
        avpu: 'U', consciousness_source: 'PARAMEDIC', chiefcomplaint: 'ARREST',
        temperature: 36.0, sbp: 80, dbp: 50, o2sat: 80
      });
    } else if (type === 'moderate') {
      setFormData({
        resprate: 32, heartrate: 110, sbp: 85, gcs_motor: 5, avpu: 'V',
        chiefcomplaint: 'Shortness of breath', temperature: 38.5, o2sat: 89
      });
    } else {
      setFormData({
        resprate: 16, heartrate: 80, sbp: 120, dbp: 80, gcs_eye: 4, gcs_verbal: 5, gcs_motor: 6,
        avpu: 'A', follows_commands: 1, chiefcomplaint: 'Fall', pain_score: 3
      });
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    
    // Clean empty strings and convert types
    const payload: Record<string, any> = {};
    for (const key in formData) {
      if (formData[key] !== '' && formData[key] !== null && formData[key] !== undefined) {
        const field = [...schema.core, ...schema.advanced].find(f => f.name === key);
        if (field?.type === 'numeric' && !isNaN(Number(formData[key]))) {
          payload[key] = Number(formData[key]);
        } else {
          payload[key] = formData[key];
        }
      }
    }
    
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    } catch (err: any) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const renderField = (f: FieldSchema) => (
    <div key={f.name} className="flex flex-col gap-1">
      <label className="text-sm font-medium text-slate-700 dark:text-slate-300">
        {f.name} {f.hint && <span className="text-xs text-slate-400">({f.hint})</span>}
      </label>
      {f.type === 'categorical' ? (
        <select 
          className="p-2 border rounded-md dark:bg-slate-800 dark:border-slate-600 focus:ring-2 focus:ring-blue-500"
          value={formData[f.name] || ''}
          onChange={e => setFormData({...formData, [f.name]: e.target.value})}
        >
          <option value="">Not measured</option>
          {f.values?.map(v => <option key={v} value={v}>{v}</option>)}
        </select>
      ) : (
        <input 
          type={f.type === 'numeric' ? 'number' : 'text'}
          step="any"
          className="p-2 border rounded-md dark:bg-slate-800 dark:border-slate-600 focus:ring-2 focus:ring-blue-500"
          placeholder="Not measured"
          value={formData[f.name] || ''}
          onChange={e => setFormData({...formData, [f.name]: e.target.value})}
        />
      )}
    </div>
  );

  const getLabelColor = (label: string) => {
    switch(label) {
      case 'RED': return 'bg-triage-red text-white';
      case 'YELLOW': return 'bg-triage-yellow text-slate-900';
      case 'GREEN': return 'bg-triage-green text-white';
      case 'BLACK': return 'bg-triage-black text-white';
      default: return 'bg-slate-200 text-slate-900';
    }
  };

  return (
    <div className="grid md:grid-cols-2 gap-8">
      {/* Form */}
      <div className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-lg font-bold">Patient Vitals & Information</h2>
          <div className="flex gap-2">
            <button onClick={() => loadExample('severe')} className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200">Ex 1: Severe</button>
            <button onClick={() => loadExample('moderate')} className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200">Ex 2: Mod</button>
            <button onClick={() => loadExample('minor')} className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200">Ex 3: Minor</button>
          </div>
        </div>
        
        <form onSubmit={handleSubmit} className="flex flex-col gap-6">
          <div className="grid grid-cols-2 gap-4">
            {schema.core.map(renderField)}
          </div>
          
          <div className="border-t border-slate-200 dark:border-slate-700 pt-4">
            <button 
              type="button" 
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-sm font-medium text-blue-600 hover:underline"
            >
              {showAdvanced ? 'Hide advanced fields' : 'Show advanced fields...'}
            </button>
            {showAdvanced && (
              <div className="grid grid-cols-2 gap-4 mt-4">
                {schema.advanced.map(renderField)}
              </div>
            )}
          </div>
          
          <p className="text-xs text-slate-500 flex gap-1 items-start">
            <Info size={14} className="shrink-0 mt-0.5" />
            Empty inputs are explicitly treated as "not measured" which is a clinically meaningful signal in this model.
          </p>
          
          <button 
            type="submit" 
            disabled={loading}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition-colors disabled:opacity-50"
          >
            {loading ? 'Processing...' : 'Triage Patient'}
          </button>
        </form>
      </div>

      {/* Results */}
      <div>
        {error && <div className="p-4 bg-red-50 text-red-600 border border-red-200 rounded-lg mb-6">{error}</div>}
        
        {result && (
          <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Main Label */}
            <div className={`p-8 rounded-xl shadow-lg flex flex-col items-center justify-center text-center ${getLabelColor(result.predicted_label)}`}>
              <div className="text-sm font-semibold opacity-90 uppercase tracking-widest mb-2">Predicted Triage Category</div>
              <div className="text-6xl font-black tracking-tight flex items-center gap-4">
                {result.predicted_label === 'RED' && <AlertCircle size={48} />}
                {result.predicted_label === 'YELLOW' && <AlertTriangle size={48} />}
                {result.predicted_label === 'GREEN' && <CheckCircle size={48} />}
                {result.predicted_label === 'BLACK' && <Skull size={48} />}
                {result.predicted_label}
              </div>
            </div>
            
            {result.differs_from_most_likely && (
              <div className="bg-yellow-50 dark:bg-yellow-900/30 border-l-4 border-yellow-400 p-4 rounded-r-lg">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  <strong>Notice:</strong> The most probable class was {result.most_likely_label}, but the model predicted {result.predicted_label} to minimize the expected misclassification cost under the official cost matrix.
                </p>
              </div>
            )}
            
            {/* START Flags */}
            {result.start_flags_fired.length > 0 && (
              <div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
                <h3 className="font-bold mb-2">START criteria detected:</h3>
                <div className="flex flex-wrap gap-2">
                  {result.start_flags_fired.map((f: string) => (
                    <span key={f} className="px-2 py-1 bg-slate-100 dark:bg-slate-700 text-xs font-mono rounded">{f}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Expected Cost Table */}
            <div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
              <h3 className="font-bold mb-4">Expected Cost of Each Choice</h3>
              <p className="text-xs text-slate-500 mb-4">Chosen because it has the lowest expected misclassification cost under the official cost matrix.</p>
              <div className="grid grid-cols-4 gap-2">
                {['RED', 'YELLOW', 'GREEN', 'BLACK'].map(l => (
                  <div key={l} className={`p-3 rounded-lg border text-center ${result.predicted_label === l ? 'border-blue-500 ring-1 ring-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-slate-200 dark:border-slate-700'}`}>
                    <div className="text-xs font-semibold mb-1">{l}</div>
                    <div className="font-mono text-lg">{result.expected_costs[l].toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Probabilities */}
            <div className="bg-white dark:bg-slate-800 p-4 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
              <h3 className="font-bold mb-4">Calibrated Probabilities</h3>
              <div className="flex flex-col gap-3">
                {['RED', 'YELLOW', 'GREEN', 'BLACK'].map(l => {
                  const prob = result.probabilities[l];
                  const pct = Math.round(prob * 100);
                  return (
                    <div key={l} className="flex items-center gap-3">
                      <div className="w-16 text-xs font-semibold text-right">{l}</div>
                      <div className="flex-grow h-3 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${getLabelColor(l).split(' ')[0]}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <div className="w-12 text-right text-xs font-mono">{pct}%</div>
                    </div>
                  );
                })}
              </div>
            </div>
            
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------
// Batch Tab
// ---------------------------
function BatchTab() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<{ preview: any[], csv_content: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_URL}/predict_batch`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }
      setResult(await res.json());
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadCsv = () => {
    if (!result) return;
    const blob = new Blob([result.csv_content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'submission_predictions.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getBadgeColor = (label: string) => {
    switch(label) {
      case 'RED': return 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300';
      case 'YELLOW': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-300';
      case 'GREEN': return 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300';
      case 'BLACK': return 'bg-slate-800 text-slate-100 dark:bg-slate-700 dark:text-slate-300';
      default: return '';
    }
  };

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-8">
      <div className="bg-white dark:bg-slate-800 p-8 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 text-center">
        <UploadCloud className="mx-auto text-blue-500 mb-4" size={48} />
        <h2 className="text-xl font-bold mb-2">Upload Test Dataset</h2>
        <p className="text-slate-500 dark:text-slate-400 mb-6 max-w-md mx-auto">
          Upload a CSV file with the same columns as the test file. The system will return a submission-ready CSV with 6 required columns.
        </p>
        
        <input type="file" accept=".csv" className="hidden" ref={fileInputRef} onChange={handleFileChange} />
        
        <div className="flex justify-center gap-4">
          <button 
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-2 border border-slate-300 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 font-medium"
          >
            {file ? file.name : 'Select CSV file'}
          </button>
          <button 
            onClick={handleUpload}
            disabled={!file || loading}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg disabled:opacity-50 flex items-center gap-2"
          >
            {loading ? 'Processing...' : 'Generate Predictions'}
          </button>
        </div>
        {error && <div className="mt-4 text-red-500 text-sm">{error}</div>}
      </div>

      {result && (
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden animate-in fade-in slide-in-from-bottom-4">
          <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center bg-slate-50 dark:bg-slate-800/50">
            <h3 className="font-bold">Preview (First 20 rows)</h3>
            <button onClick={downloadCsv} className="flex items-center gap-2 text-sm font-bold text-blue-600 hover:text-blue-700">
              <Download size={16} /> Download Full CSV
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 font-semibold uppercase text-xs">
                <tr>
                  <th className="px-4 py-3">Patient ID</th>
                  <th className="px-4 py-3">Predicted Triage</th>
                  <th className="px-4 py-3 text-right">RED Prob</th>
                  <th className="px-4 py-3 text-right">YELLOW Prob</th>
                  <th className="px-4 py-3 text-right">GREEN Prob</th>
                  <th className="px-4 py-3 text-right">BLACK Prob</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {result.preview.map((row, i) => (
                  <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                    <td className="px-4 py-2 font-mono text-xs">{row['Patient ID']}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold ${getBadgeColor(row['Predicted Triage'])}`}>
                        {row['Predicted Triage']}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs text-slate-500">{row['RED Probability'].toFixed(4)}</td>
                    <td className="px-4 py-2 text-right font-mono text-xs text-slate-500">{row['YELLOW Probability'].toFixed(4)}</td>
                    <td className="px-4 py-2 text-right font-mono text-xs text-slate-500">{row['GREEN Probability'].toFixed(4)}</td>
                    <td className="px-4 py-2 text-right font-mono text-xs text-slate-500">{row['BLACK Probability'].toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------
// Report Tab
// ---------------------------
function ReportTab() {
  const [report, setReport] = useState<any>(null);
  const [error, setError] = useState<string>('');
  
  useEffect(() => {
    fetch(`${API_URL}/report`)
      .then(r => {
         if (!r.ok) throw new Error("API Error " + r.status);
         return r.json();
      })
      .then(data => {
         if (data.detail) throw new Error(data.detail);
         setReport(data);
      })
      .catch(e => setError(e.toString()));
  }, []);

  if (error) return <div className="text-center py-12 text-red-500 font-bold">Error loading report: {error}</div>;
  if (!report) return <div className="text-center py-12">Loading report...</div>;

  try {
    return (
      <div className="flex flex-col gap-8 max-w-5xl mx-auto">
        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-xl border border-blue-200 dark:border-blue-800 text-sm flex gap-3">
          <Info className="shrink-0 text-blue-500" />
          <div>
            <p>These metrics represent <strong>cross-validated out-of-fold estimates</strong> on the training data. They evaluate how the model might perform on unseen patients from the same distribution, but do not represent the final hidden test score.</p>
          </div>
        </div>
        
        {/* Overview Table */}
        <div className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
          <h2 className="text-xl font-bold mb-4">Decision Rule Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 font-semibold uppercase text-xs">
                <tr>
                  <th className="px-4 py-3">Approach</th>
                  <th className="px-4 py-3">Accuracy</th>
                  <th className="px-4 py-3">Macro-F1</th>
                  <th className="px-4 py-3">Total Cost</th>
                  <th className="px-4 py-3">RED as GREEN</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {report.comparison && Object.entries(report.comparison).map(([key, metrics]: [string, any]) => (
                  <tr key={key} className={key === 'CalEns_minrisk' ? 'bg-green-50 dark:bg-green-900/20 font-medium' : ''}>
                    <td className="px-4 py-3">{key.replace('_', ' ')} {key === 'CalEns_minrisk' && '(Final)'}</td>
                    <td className="px-4 py-3">{Number(metrics.Accuracy || 0).toFixed(3)}</td>
                    <td className="px-4 py-3">{Number(metrics['Macro-F1'] || 0).toFixed(3)}</td>
                    <td className="px-4 py-3 font-mono font-bold text-red-600 dark:text-red-400">{metrics.Cost}</td>
                    <td className="px-4 py-3">{metrics.R_as_G}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8">
          {/* Classification Report */}
          <div className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
            <h2 className="text-xl font-bold mb-4">Per-class Metrics (Final)</h2>
            <table className="w-full text-sm text-right">
              <thead className="text-slate-500 border-b">
                <tr>
                  <th className="text-left pb-2">Class</th>
                  <th className="pb-2">Precision</th>
                  <th className="pb-2">Recall</th>
                  <th className="pb-2">F1</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {report.per_class && ['RED', 'YELLOW', 'GREEN', 'BLACK'].map(l => (
                  <tr key={l}>
                    <td className="py-2 text-left font-bold">{l}</td>
                    <td className="py-2">{Number(report.per_class[l]?.precision || 0).toFixed(3)}</td>
                    <td className="py-2">{Number(report.per_class[l]?.recall || 0).toFixed(3)}</td>
                    <td className="py-2">{Number(report.per_class[l]?.['f1-score'] || 0).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Confusion Matrix */}
          <div className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
            <h2 className="text-xl font-bold mb-4">Confusion Matrix</h2>
            <img src={`${API_URL}/figures/confusion_matrix.png`} alt="Confusion Matrix" className="w-full h-auto rounded-lg" />
          </div>
          
          {/* Top Features */}
          <div className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 md:col-span-2">
            <h2 className="text-xl font-bold mb-4">Top 10 Important Features (LightGBM)</h2>
            <div className="flex flex-wrap gap-2">
              {report.top_features && Object.entries(report.top_features).map(([f, imp]: [string, any]) => (
                <div key={f} className="px-3 py-2 bg-slate-100 dark:bg-slate-700 rounded-lg text-sm flex gap-2 justify-between items-center w-full sm:w-auto sm:flex-grow">
                  <span className="font-mono text-slate-700 dark:text-slate-300">{f}</span>
                  <span className="font-bold">{Number(imp || 0).toFixed(1)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  } catch (err: any) {
    return <div className="text-center py-12 text-red-500 font-bold">Render Error: {err.message}</div>;
  }
}

// ---------------------------
// About Tab
// ---------------------------
function AboutTab() {
  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-8 text-slate-800 dark:text-slate-200 leading-relaxed">
      <div className="bg-white dark:bg-slate-800 p-8 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
        <h2 className="text-2xl font-bold mb-4">How It Works</h2>
        <p className="mb-6">The ML pipeline leverages a three-model ensemble with specialized calibration to prioritize patient safety using a cost-sensitive decision rule.</p>
        
        <div className="flex flex-col gap-4 bg-slate-50 dark:bg-slate-900 p-6 rounded-xl border border-slate-200 dark:border-slate-700 mb-8">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">1</div>
            <div><strong>Feature Engineering:</strong> Add missingness indicators and boolean flags based on START triage criteria.</div>
          </div>
          <div className="w-0.5 h-4 bg-slate-300 ml-4"></div>
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">2</div>
            <div><strong>Ensemble:</strong> Average the predicted probabilities from LightGBM, ExtraTrees, and Logistic Regression.</div>
          </div>
          <div className="w-0.5 h-4 bg-slate-300 ml-4"></div>
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">3</div>
            <div><strong>Calibration:</strong> Apply multinomial logistic regression to fix over/under confidence.</div>
          </div>
          <div className="w-0.5 h-4 bg-slate-300 ml-4"></div>
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">4</div>
            <div><strong>Minimum-Risk Decision:</strong> Select the category that minimizes expected cost.</div>
          </div>
        </div>
        
        <h3 className="text-xl font-bold mb-4">Expected Cost Formula</h3>
        <p className="font-mono bg-slate-100 dark:bg-slate-700 p-4 rounded-lg mb-8 text-center text-lg">
          Cost(j) = &sum; P(i) &times; CostMatrix[i][j]
        </p>
        
        <h3 className="text-xl font-bold mb-4">Official Cost Matrix</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-center border-collapse">
            <thead>
              <tr className="bg-slate-100 dark:bg-slate-700">
                <th className="p-3 border border-slate-300 dark:border-slate-600">Actual \ Pred</th>
                <th className="p-3 border border-slate-300 dark:border-slate-600 font-bold text-red-600 dark:text-red-400">RED</th>
                <th className="p-3 border border-slate-300 dark:border-slate-600 font-bold text-yellow-600 dark:text-yellow-400">YELLOW</th>
                <th className="p-3 border border-slate-300 dark:border-slate-600 font-bold text-green-600 dark:text-green-400">GREEN</th>
                <th className="p-3 border border-slate-300 dark:border-slate-600 font-bold text-slate-800 dark:text-slate-300">BLACK</th>
              </tr>
            </thead>
            <tbody>
              <tr><td className="p-3 border font-bold text-left">RED</td><td className="p-3 border font-bold">0</td><td className="p-3 border">5</td><td className="p-3 border text-red-600">10</td><td className="p-3 border">5</td></tr>
              <tr><td className="p-3 border font-bold text-left">YELLOW</td><td className="p-3 border">2</td><td className="p-3 border font-bold">0</td><td className="p-3 border">3</td><td className="p-3 border">2</td></tr>
              <tr><td className="p-3 border font-bold text-left">GREEN</td><td className="p-3 border">1</td><td className="p-3 border">1</td><td className="p-3 border font-bold">0</td><td className="p-3 border">1</td></tr>
              <tr><td className="p-3 border font-bold text-left">BLACK</td><td className="p-3 border">2</td><td className="p-3 border">2</td><td className="p-3 border">2</td><td className="p-3 border font-bold">0</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default App;
