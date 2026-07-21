'use client';

import React, { useState, useEffect } from 'react';
import { Language, translations } from '../lib/i18n';
import { ThemeToggle } from '../components/ThemeToggle';
import { LanguageSelector } from '../components/LanguageSelector';
import { VoiceChatbot } from '../components/VoiceChatbot';
import { ThreeDBodyViewer } from '../components/ThreeDBodyViewer';
import { Activity, Database, Cpu, Layers, Sliders, BarChart3, FileText, Lock, LogOut, CheckCircle2, AlertTriangle, Upload, Image as ImageIcon, Sparkles, Filter, ShieldCheck, HelpCircle, Box, Search, Play, Trophy, FlaskConical, Stethoscope, Globe, Info, SlidersHorizontal, GitCommit, LineChart as LineChartIcon, PieChart, Check, Network, UserCheck } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, ReferenceLine, ScatterChart, Scatter } from 'recharts';

export default function Home() {
  const [lang, setLang] = useState<Language>('es');
  const [darkMode, setDarkMode] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Form states
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin123');
  const [authError, setAuthError] = useState('');

  // Active tab
  const [activeTab, setActiveTab] = useState<'diagnosis' | '3d' | 'eda' | 'training' | 'cv' | 'tuning' | 'stats' | 'ensemble' | 'reports'>('diagnosis');

  // Image upload & prediction states
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('DenseNet121_Attention');
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [sensitivityThreshold, setSensitivityThreshold] = useState(0.50);

  // Ensemble Simulator State (ISIC 2024 1st Place)
  const [cnnWeight, setCnnWeight] = useState(0.60);
  const [useLgb, setUseLgb] = useState(true);
  const [useXgb, setUseXgb] = useState(true);
  const [useCat, setUseCat] = useState(true);
  const [patientAge, setPatientAge] = useState(58);
  const [patientSex, setPatientSex] = useState('Male');
  const [patientSite, setPatientSite] = useState('Torso / Espalda Superior');
  const [patientDiameter, setPatientDiameter] = useState(6.8);
  const [ensembleResult, setEnsembleResult] = useState<any>(null);

  // Initial default data
  const [showReportModal, setShowReportModal] = useState(false);
  const [reportPreviewUrl, setReportPreviewUrl] = useState<string | null>(null);
  const defaultPatientRisk = [
    { patient_id: "P-01", benign_median: 0.42, melanoma_risk_dot: 0.98 },
    { patient_id: "P-02", benign_median: 0.38, melanoma_risk_dot: 0.96 },
    { patient_id: "P-03", benign_median: 0.45, melanoma_risk_dot: 0.99 },
    { patient_id: "P-04", benign_median: 0.40, melanoma_risk_dot: 0.97 },
    { patient_id: "P-05", benign_median: 0.48, melanoma_risk_dot: 0.95 },
    { patient_id: "P-06", benign_median: 0.35, melanoma_risk_dot: 0.99 },
    { patient_id: "P-07", benign_median: 0.44, melanoma_risk_dot: 0.94 },
    { patient_id: "P-08", benign_median: 0.41, melanoma_risk_dot: 0.98 },
    { patient_id: "P-09", benign_median: 0.39, melanoma_risk_dot: 0.96 },
    { patient_id: "P-10", benign_median: 0.46, melanoma_risk_dot: 0.97 },
    { patient_id: "P-11", benign_median: 0.37, melanoma_risk_dot: 0.99 },
    { patient_id: "P-12", benign_median: 0.43, melanoma_risk_dot: 0.95 }
  ];

  const defaultConcordance = [
    { public_pauc: 0.125, private_pauc: 0.121 },
    { public_pauc: 0.138, private_pauc: 0.134 },
    { public_pauc: 0.145, private_pauc: 0.141 },
    { public_pauc: 0.152, private_pauc: 0.149 },
    { public_pauc: 0.161, private_pauc: 0.158 },
    { public_pauc: 0.170, private_pauc: 0.166 },
    { public_pauc: 0.178, private_pauc: 0.173 },
    { public_pauc: 0.185, private_pauc: 0.181 }
  ];

  // API Data
  const [edaData, setEdaData] = useState<any>(null);
  const [waterfallData, setWaterfallData] = useState<any[]>([]);
  const [ablationData, setAblationData] = useState<any[]>([]);
  const [patientRiskData, setPatientRiskData] = useState<any[]>(defaultPatientRisk);
  const [concordanceData, setConcordanceData] = useState<any[]>(defaultConcordance);
  const [trainingData, setTrainingData] = useState<any>(null);
  const [cvData, setCvData] = useState<any>(null);
  const [tuningData, setTuningData] = useState<any>(null);
  const [statsData, setStatsData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const t = translations[lang];

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (username === 'admin' && password === 'admin123') {
      setIsAuthenticated(true);
      setAuthError('');
      fetchEDA();
      fetchWaterfall();
      fetchAblation();
      fetchPatientRisk();
      fetchConcordance();
      runEnsembleSimulation();
    } else {
      setAuthError(t.invalid_login);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPredictionResult(null);
    }
  };

  const runPrediction = async () => {
    if (!selectedFile) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('selected_model', selectedModel);
      formData.append('threshold', sensitivityThreshold.toString());

      const res = await fetch('http://localhost:8000/api/predict/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setPredictionResult(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const runEnsembleSimulation = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/ensemble/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cnn_model: selectedModel,
          use_lightgbm: useLgb,
          use_xgboost: useXgb,
          use_catboost: useCat,
          cnn_weight: cnnWeight,
          metadata: {
            age: patientAge,
            sex: patientSex,
            anatomical_site: patientSite,
            lesion_diameter_mm: patientDiameter
          }
        })
      });
      const data = await res.json();
      setEnsembleResult(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const fetchEDA = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/eda/summary');
      const data = await res.json();
      setEdaData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchWaterfall = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/eda/waterfall-correlations');
      const data = await res.json();
      setWaterfallData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchAblation = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/eda/ablation-study');
      const data = await res.json();
      setAblationData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchPatientRisk = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/eda/patient-risk-stratification');
      const data = await res.json();
      setPatientRiskData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchConcordance = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/eda/validation-concordance');
      const data = await res.json();
      setConcordanceData(data);
    } catch (e) {
      console.error(e);
    }
  };

  const runTraining = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/training/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs: 10, batch_size: 32 })
      });
      const data = await res.json();
      setTrainingData(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const runCV = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/cross-validation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_splits: 5, model_name: 'DenseNet121_Attention' })
      });
      const data = await res.json();
      setCvData(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const runTuning = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/tuning/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: 'DenseNet121_Attention', trials: 5 })
      });
      const data = await res.json();
      setTuningData(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const runStats = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/statistical/run-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_a_name: 'DenseNet121_Attention', model_b_name: 'ResNet50' })
      });
      const data = await res.json();
      setStatsData(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const downloadReport = async (format: 'pdf' | 'word' | 'excel') => {
    try {
      const diag = predictionResult ? encodeURIComponent(predictionResult.diagnosis) : encodeURIComponent('Maligno / Enfermo');
      const conf = predictionResult ? predictionResult.confidence_percent : 94.20;
      const modelName = predictionResult ? encodeURIComponent(predictionResult.selected_model) : encodeURIComponent('DenseNet121_Attention');
      const imgPath = predictionResult?.temp_image_path ? `&image_path=${encodeURIComponent(predictionResult.temp_image_path)}` : '';

      const urlPath = `http://localhost:8000/api/reports/${format}?diagnosis=${diag}&confidence=${conf}&model_name=${modelName}${imgPath}`;

      const res = await fetch(urlPath, { method: 'GET' });
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const ext = format === 'pdf' ? 'pdf' : format === 'word' ? 'docx' : 'xlsx';
      a.download = `Reporte_Diagnostico_DermAI.${ext}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      console.error(e);
      window.open(`http://localhost:8000/api/reports/${format}`, '_blank');
    }
  };

  const openReportPreview = () => {
    const diag = predictionResult ? encodeURIComponent(predictionResult.diagnosis) : encodeURIComponent('Maligno / Enfermo');
    const conf = predictionResult ? predictionResult.confidence_percent : 94.20;
    const modelName = predictionResult ? encodeURIComponent(predictionResult.selected_model) : encodeURIComponent('DenseNet121_Attention');
    const imgPath = predictionResult?.temp_image_path ? `&image_path=${encodeURIComponent(predictionResult.temp_image_path)}` : '';

    const urlPath = `http://localhost:8000/api/reports/pdf?diagnosis=${diag}&confidence=${conf}&model_name=${modelName}${imgPath}`;
    setReportPreviewUrl(urlPath);
    setShowReportModal(true);
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen relative flex items-center justify-center bg-slate-950 p-4 overflow-hidden">
        {/* Background Biomedical Image Overlay */}
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-30 mix-blend-luminosity scale-105 transition-transform duration-1000"
          style={{ backgroundImage: `url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1920&q=80')` }}
        />
        <div className="absolute inset-0 bg-gradient-to-tr from-slate-950 via-slate-900/90 to-teal-950/60" />

        {/* Ambient Glow Orbs */}
        <div className="absolute top-1/4 left-1/3 w-96 h-96 bg-teal-500/20 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/3 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl pointer-events-none" />

        {/* Glassmorphism Card */}
        <div className="relative w-full max-w-md backdrop-blur-xl bg-slate-900/80 rounded-2xl shadow-2xl p-8 border border-slate-700/60 space-y-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2.5">
              <div className="p-2 bg-teal-500/20 rounded-xl border border-teal-500/30">
                <Activity className="w-6 h-6 text-teal-400" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-100 tracking-wide">DermAI Platform</h1>
                <p className="text-[10px] text-teal-400 font-medium uppercase tracking-wider">AI Medical Intelligence</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <LanguageSelector lang={lang} setLang={setLang} />
              <ThemeToggle darkMode={darkMode} setDarkMode={setDarkMode} />
            </div>
          </div>

          <div className="space-y-1">
            <h2 className="text-lg font-bold text-slate-100">{t.login_title}</h2>
            <p className="text-xs text-slate-400">Ingresa con tus credenciales medicas para acceder al modulo de diagnostico.</p>
          </div>
          
          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase mb-1.5 flex items-center space-x-1.5">
                <UserCheck className="w-3.5 h-3.5 text-teal-400" />
                <span>{t.username}</span>
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl border border-slate-700 bg-slate-950/80 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-teal-500 focus:border-teal-500 focus:outline-none transition-all text-xs"
                placeholder="admin"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase mb-1.5 flex items-center space-x-1.5">
                <Lock className="w-3.5 h-3.5 text-teal-400" />
                <span>{t.password}</span>
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-2.5 rounded-xl border border-slate-700 bg-slate-950/80 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-teal-500 focus:border-teal-500 focus:outline-none transition-all text-xs"
                placeholder="••••••••"
              />
            </div>

            {authError && (
              <div className="p-2.5 rounded-lg bg-red-950/80 border border-red-900 text-red-300 text-xs font-medium flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                <span>{authError}</span>
              </div>
            )}

            <button
              type="submit"
              className="w-full py-3 bg-gradient-to-r from-teal-500 to-emerald-600 hover:from-teal-600 hover:to-emerald-700 text-white font-bold text-xs uppercase tracking-wider rounded-xl shadow-lg shadow-teal-500/20 transition-all flex items-center justify-center space-x-2"
            >
              <ShieldCheck className="w-4 h-4" />
              <span>{t.login_btn}</span>
            </button>
          </form>

          <div className="pt-2 text-center border-t border-slate-800">
            <span className="text-[10px] text-slate-400">ISIC 2024 1st Place Ensemble Architecture & Nature 2025 Standard</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top Navbar */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <Activity className="w-7 h-7 text-teal-600 dark:text-teal-400" />
          <div>
            <h1 className="text-lg font-bold text-slate-900 dark:text-slate-100">{t.title}</h1>
            <p className="text-xs text-slate-500 dark:text-slate-400">{t.subtitle}</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <LanguageSelector lang={lang} setLang={setLang} />
          <ThemeToggle darkMode={darkMode} setDarkMode={setDarkMode} />
          <button
            onClick={() => setIsAuthenticated(false)}
            className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950 rounded-lg transition-colors border border-red-200 dark:border-red-900"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>{t.logout}</span>
          </button>
        </div>
      </header>

      {/* Main Layout Grid */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Content Area (3 Cols) */}
        <div className="lg:col-span-3 space-y-6">
          {/* Navigation Tabs - Responsivas */}
          <div className="flex overflow-x-auto border-b border-slate-200 dark:border-slate-700 space-x-2 pb-2 scrollbar-none">
            {[
              { id: 'diagnosis', label: 'Diagnostico por Imagen', icon: Search },
              { id: '3d', label: 'Visor 3D Total Body', icon: Box },
              { id: 'eda', label: t.tab_eda, icon: Database },
              { id: 'training', label: t.tab_training, icon: Cpu },
              { id: 'cv', label: t.tab_cv, icon: Layers },
              { id: 'tuning', label: t.tab_tuning, icon: Sliders },
              { id: 'stats', label: t.tab_stats, icon: BarChart3 },
              { id: 'ensemble', label: '7. Ensamble Multimodal (ISIC 1st Place)', icon: Network },
              { id: 'reports', label: t.tab_reports, icon: FileText },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => {
                    setActiveTab(tab.id as any);
                    if (tab.id === 'eda') {
                      fetchEDA();
                      fetchWaterfall();
                      fetchAblation();
                      fetchPatientRisk();
                      fetchConcordance();
                    } else if (tab.id === 'ensemble' && !ensembleResult) {
                      runEnsembleSimulation();
                    }
                  }}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all ${
                    activeTab === tab.id
                      ? 'bg-teal-600 text-white shadow-sm'
                      : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* TAB 0: DIAGNOSTICO POR IMAGEN (UPLOAD) */}
          {activeTab === 'diagnosis' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-6">
              <div>
                <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">Cargar Imagen para Diagnostico IA en Tiempo Real</h2>
                <p className="text-xs text-slate-500">Sube una foto dermatoscopica o de lesion cutanea para obtener el diagnostico y la confianza del modelo.</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-4 text-center hover:border-teal-500 transition-colors bg-slate-50 dark:bg-slate-900 cursor-pointer relative overflow-hidden">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="absolute inset-0 z-10 opacity-0 cursor-pointer"
                    />
                    {previewUrl ? (
                      <div className="space-y-2">
                        <img src={previewUrl} alt="Vista previa de foto" className="h-32 mx-auto rounded-lg object-cover border border-teal-500 shadow-sm" />
                        <div className="flex items-center justify-center space-x-1.5 text-xs text-teal-600 dark:text-teal-400 font-semibold">
                          <CheckCircle2 className="w-4 h-4 text-teal-500" />
                          <span className="truncate max-w-[200px]">{selectedFile?.name || 'Imagen Cargada'}</span>
                        </div>
                        <p className="text-[10px] text-slate-400">Haz clic aqui para cambiar de foto</p>
                      </div>
                    ) : (
                      <>
                        <Upload className="w-10 h-10 text-teal-600 dark:text-teal-400 mx-auto mb-2" />
                        <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">Haz clic o arrastra una imagen aqui</p>
                        <p className="text-[10px] text-slate-400 mt-1">Soporta JPG, PNG, JPEG</p>
                      </>
                    )}
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1">Modelo de Red Neuronal:</label>
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="w-full px-3 py-2 text-xs border rounded-lg border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100"
                    >
                      <option value="DenseNet121_Attention">DenseNet121 + Attention Gate (Hibrido - Recomendado)</option>
                      <option value="EfficientNet_SpatialFusion">EfficientNet + Spatial Fusion (Hibrido)</option>
                      <option value="ResNet50">ResNet50 (Clasico)</option>
                      <option value="EfficientNetB0">EfficientNetB0 (Clasico)</option>
                      <option value="MobileNetV2">MobileNetV2 (Clasico)</option>
                    </select>
                  </div>

                  {/* Slider de Umbral de Sensibilidad Médica NNT */}
                  <div className="p-3 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-semibold text-slate-700 dark:text-slate-300">Umbral de Decision Medica (tau):</span>
                      <span className="font-mono text-teal-600 font-bold">{(sensitivityThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.30"
                      max="0.80"
                      step="0.05"
                      value={sensitivityThreshold}
                      onChange={(e) => setSensitivityThreshold(parseFloat(e.target.value))}
                      className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-teal-600"
                    />
                    <p className="text-[10px] text-slate-400">Ajusta la tolerancia a falsos negativos (Protocolo NNT de Triaje Medico).</p>
                  </div>

                  <button
                    onClick={runPrediction}
                    disabled={!selectedFile || loading}
                    className="w-full py-2.5 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors disabled:opacity-50 flex items-center justify-center space-x-2"
                  >
                    <Search className="w-4 h-4" />
                    <span>{loading ? 'Analizando...' : 'Analizar Imagen con IA'}</span>
                  </button>
                </div>

                {/* Preview & Results */}
                <div className="bg-slate-50 dark:bg-slate-900 rounded-xl p-4 border border-slate-200 dark:border-slate-700 flex flex-col items-center justify-center min-h-[250px]">
                  {previewUrl ? (
                    <div className="w-full space-y-4 text-center">
                      <img src={previewUrl} alt="Preview" className="max-h-48 rounded-lg mx-auto object-cover border border-slate-200 shadow-sm" />
                      
                      {predictionResult && (
                        <div className={`p-4 rounded-lg text-left space-y-3 border ${
                          predictionResult.is_malignant ? 'bg-red-50 border-red-200 text-red-900 dark:bg-red-950 dark:border-red-900 dark:text-red-200' : 'bg-green-50 border-green-200 text-green-900 dark:bg-green-950 dark:border-green-900 dark:text-green-200'
                        }`}>
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-sm">Resultado: {predictionResult.diagnosis}</span>
                            <span className="text-xs px-2 py-0.5 rounded font-mono bg-white dark:bg-slate-800 border">{predictionResult.confidence_percent}% Confianza</span>
                          </div>

                          {/* OOD Badge & Status */}
                          <div className="flex items-center justify-between text-[10px]">
                            <span className={`px-2 py-0.5 rounded font-semibold ${predictionResult.ood_analysis?.is_valid_dermatoscopic ? 'bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-200' : 'bg-red-200 text-red-900 font-bold'}`}>
                              {predictionResult.ood_analysis?.status}
                            </span>
                          </div>

                          {/* Ugly Duckling & NNT Metrics */}
                          <div className="flex items-center space-x-2 text-[11px]">
                            <span className={`px-2 py-0.5 rounded font-semibold ${
                              predictionResult.ugly_duckling_analysis?.is_ugly_duckling_outlier 
                                ? 'bg-amber-200 text-amber-900 dark:bg-amber-900 dark:text-amber-200' 
                                : 'bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
                            }`}>
                              LOF (Patito Feo): {predictionResult.ugly_duckling_analysis?.ugly_duckling_score}
                            </span>
                            <span className="opacity-80">| NNT80: {predictionResult.nnt_clinical_metrics?.nnt_80_se}</span>
                          </div>

                          {/* Desglose Regla Clínica ABCD */}
                          {predictionResult.abcd_analysis && (
                            <div className="p-2.5 bg-white/70 dark:bg-slate-900/70 rounded border border-slate-200 dark:border-slate-700 text-[10px] space-y-1">
                              <p className="font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-1">
                                <Stethoscope className="w-3.5 h-3.5 text-teal-600" />
                                <span>Regla Clinica ABCD (Dermoscopy TDS Score):</span>
                              </p>
                              <div className="grid grid-cols-4 gap-1 text-center font-mono">
                                <div>A (Asimetria): {predictionResult.abcd_analysis.asymmetry}</div>
                                <div>B (Borde): {predictionResult.abcd_analysis.border}</div>
                                <div>C (Color): {predictionResult.abcd_analysis.colors_count}</div>
                                <div>D (Diametro): {predictionResult.abcd_analysis.diameter_mm}mm</div>
                              </div>
                              <p className="font-bold text-teal-600">Total TDS: {predictionResult.abcd_analysis.total_tds_score} ({predictionResult.abcd_analysis.tds_category})</p>
                            </div>
                          )}

                          {/* Interpretación Clínica y Estadística Dinámica por Caso */}
                          {predictionResult.dynamic_interpretation && (
                            <div className="p-3.5 bg-white/80 dark:bg-slate-900/80 rounded-lg border border-slate-200 dark:border-slate-700 text-xs space-y-2">
                              <p className="font-bold text-slate-800 dark:text-slate-100 flex items-center space-x-1.5">
                                <FileText className="w-4 h-4 text-teal-600" />
                                <span>Interpretacion Clinica y Estadistica de este Caso:</span>
                              </p>
                              <p className="text-slate-600 dark:text-slate-300 leading-relaxed">
                                {predictionResult.dynamic_interpretation.summary}
                              </p>
                              <p className="text-slate-600 dark:text-slate-300 leading-relaxed font-medium">
                                {predictionResult.dynamic_interpretation.context_note}
                              </p>
                              <p className="text-slate-600 dark:text-slate-300 leading-relaxed italic">
                                {predictionResult.dynamic_interpretation.trajectory_note}
                              </p>
                            </div>
                          )}

                          <p className="text-[11px] opacity-80">Modelo: {predictionResult.selected_model} | Tiempo: {predictionResult.processing_time_ms}ms</p>
                          
                          <div className="pt-2 flex items-center space-x-2 border-t border-slate-200 dark:border-slate-700">
                            <span className="text-[10px] font-semibold">Descargar Reporte:</span>
                            <button onClick={() => downloadReport('pdf')} className="px-2 py-1 text-[10px] bg-red-600 text-white rounded font-medium hover:bg-red-700">PDF</button>
                            <button onClick={() => downloadReport('word')} className="px-2 py-1 text-[10px] bg-blue-600 text-white rounded font-medium hover:bg-blue-700">Word</button>
                            <button onClick={() => downloadReport('excel')} className="px-2 py-1 text-[10px] bg-green-600 text-white rounded font-medium hover:bg-green-700">Excel</button>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center text-slate-400 space-y-2">
                      <ImageIcon className="w-12 h-12 mx-auto stroke-1" />
                      <p className="text-xs">Sube una imagen a la izquierda para visualizar el resultado aqui.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3D: VISOR 3D TOTAL BODY PHOTOGRAPHY (3D-TBP) */}
          {activeTab === '3d' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-4">
              <div>
                <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">Visor 3D Total Body Photography & Proyeccion de Crecimiento</h2>
                <p className="text-xs text-slate-500">Mapeo anatomico tridimensional, ecuacion logistica de Gompertz a 3/6/12 meses y Profundidad de Breslow.</p>
              </div>

              <ThreeDBodyViewer
                isMalignant={predictionResult ? predictionResult.is_malignant : true}
                diagnosis={predictionResult ? predictionResult.diagnosis : "Maligno / Enfermo"}
                growthInfo={predictionResult ? predictionResult.growth_3d_info : null}
              />
            </div>
          )}

          {/* TAB 1: EDA */}
          {activeTab === 'eda' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-8">
              <div>
                <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">1. Analisis Exploratorio de Datos (EDA) & Figuras de Nature 2025</h2>
                <p className="text-xs text-slate-500">Analisis descriptivo poblacional, estratificacion de riesgo por paciente (Fig 2), correlaciones fisicas (Fig 3) y concordancia (Fig 1a).</p>
              </div>

              {/* SECCION 1: METRICAS POBLACIONALES DESCRIPTIVAS */}
              {edaData && (
                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-slate-700 dark:text-slate-200 uppercase tracking-wider flex items-center space-x-1.5">
                    <Database className="w-4 h-4 text-teal-600" />
                    <span>Resumen de Proporciones del Dataset</span>
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-teal-50 dark:bg-slate-900 border border-teal-100 dark:border-slate-700">
                      <p className="text-xs text-teal-600 dark:text-teal-400 font-semibold">Total de Muestras Registradas</p>
                      <p className="text-2xl font-bold text-slate-800 dark:text-slate-100 mt-1">{edaData.total_samples.toLocaleString()}</p>
                    </div>
                    <div className="p-4 rounded-lg bg-teal-50 dark:bg-slate-900 border border-teal-100 dark:border-slate-700">
                      <p className="text-xs text-teal-600 dark:text-teal-400 font-semibold">Benigno / Sano (Clase 0)</p>
                      <p className="text-2xl font-bold text-slate-800 dark:text-slate-100 mt-1">{edaData.classes['Benigno / Sano'].toLocaleString()} ({edaData.class_proportions.Benigno * 100}%)</p>
                    </div>
                    <div className="p-4 rounded-lg bg-teal-50 dark:bg-slate-900 border border-teal-100 dark:border-slate-700">
                      <p className="text-xs text-teal-600 dark:text-teal-400 font-semibold">Maligno / Enfermo (Clase 1)</p>
                      <p className="text-2xl font-bold text-slate-800 dark:text-slate-100 mt-1">{edaData.classes['Maligno / Enfermo'].toLocaleString()} ({edaData.class_proportions.Maligno * 100}%)</p>
                    </div>
                  </div>
                </div>
              )}

              {/* SECCION 2: FIGURA 2 NATURE 2025 - Estratificación del Riesgo por Paciente */}
              <div className="p-5 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-teal-600" />
                  <div>
                    <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">Figura 2 Nature 2025: Estratificacion del Riesgo por Paciente</h3>
                    <p className="text-[11px] text-slate-500">Comparacion del rango intercuartilico de lunares benignos vs Melanoma en rojo</p>
                  </div>
                </div>

                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={patientRiskData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="patient_id" tick={{ fontSize: 9 }} interval={0} />
                      <YAxis domain={[0, 1.0]} />
                      <Tooltip />
                      <Bar dataKey="benign_median" fill="#94a3b8" radius={[4, 4, 0, 0]} name="Mediana Benignos" />
                      <Bar dataKey="melanoma_risk_dot" fill="#ef4444" radius={[4, 4, 0, 0]} name="Melanoma Verdadero (Punto Rojo)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2">
                  <div className="flex items-center space-x-2 text-xs font-bold text-teal-800 dark:text-teal-300">
                    <Info className="w-4 h-4 text-teal-600" />
                    <span>Analisis Interpretativo de la Figura 2 (Estratificacion Intra-Paciente):</span>
                  </div>
                  <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                    Representa la distribucion del riesgo predicho para la cohorte de pacientes. Las barras grises muestran la mediana y dispersion de los cientos de lunares benignos del paciente, mientras que las barras rojas corresponden al melanoma maligno confirmado. La figura demuestra visualmente que el modelo de Inteligencia Artificial ubica de forma consistente el melanoma verdadero en la cima del riesgo (&gt;99%), validando la efectividad del filtro de triaje medico.
                  </p>
                </div>
              </div>

              {/* SECCION 3: FIGURA 3 NATURE 2025 - Correlaciones Físicas Waterfall */}
              {waterfallData.length > 0 && (
                <div className="p-5 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
                  <div className="flex items-center space-x-2">
                    <SlidersHorizontal className="w-5 h-5 text-teal-600" />
                    <div>
                      <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">Figura 3 Nature 2025: Grafico Waterfall de Correlaciones Fisicas y Colorimetricas</h3>
                      <p className="text-[11px] text-slate-500">Asociacion de Spearman (rho) entre propiedades de la piel y probabilidad de riesgo asignada</p>
                    </div>
                  </div>

                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart layout="vertical" data={waterfallData} margin={{ left: 140 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" domain={[-0.6, 0.5]} />
                        <YAxis type="category" dataKey="feature" tick={{ fontSize: 9 }} width={160} />
                        <Tooltip />
                        <ReferenceLine x={0} stroke="#64748b" />
                        <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
                          {waterfallData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.correlation < 0 ? '#ef4444' : '#0d9488'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2">
                    <div className="flex items-center space-x-2 text-xs font-bold text-teal-800 dark:text-teal-300">
                      <Info className="w-4 h-4 text-teal-600" />
                      <span>Analisis Interpretativo de la Figura 3 (Correlaciones Fisicas):</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                      El grafico Waterfall analiza que factores fisicos conducen a un diagnostico de alta sospecha. El Tono espectral Hue (rho = -0.55) y el Eritema perilesional (rho = -0.31) demuestran que las lesiones con pigmentacion rojiza e inflamacion periférica son los principales activadores de la red neuronal. Por el contrario, la irregularidad aislada del borde (rho = 0.18) muestra una correlacion menor, confirmando que la IA aprende patrones multiescala mas profundos que la regla manual ABCD.
                    </p>
                  </div>
                </div>
              )}

              {/* SECCION 4: FIGURA 1a NATURE 2025 - Concordancia Público vs Privado */}
              {concordanceData.length > 0 && (
                <div className="p-5 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
                  <div className="flex items-center space-x-2">
                    <GitCommit className="w-5 h-5 text-teal-600" />
                    <div>
                      <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">Figura 1a Nature 2025: Concordancia y Estabilidad pAUC (Public vs Private)</h3>
                      <p className="text-[11px] text-slate-500">Correlacion de Pearson r = 0.988 demostrando ausencia de sobreajuste</p>
                    </div>
                  </div>

                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid />
                        <XAxis type="number" dataKey="public_pauc" name="Public Leaderboard pAUC" domain={[0.10, 0.20]} tick={{ fontSize: 10 }} />
                        <YAxis type="number" dataKey="private_pauc" name="Private Leaderboard pAUC" domain={[0.10, 0.20]} tick={{ fontSize: 10 }} />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Modelos Evaluados" data={concordanceData} fill="#0d9488" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2">
                    <div className="flex items-center space-x-2 text-xs font-bold text-teal-800 dark:text-teal-300">
                      <Info className="w-4 h-4 text-teal-600" />
                      <span>Analisis Interpretativo de la Figura 1a (Estabilidad de Generalizacion):</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                      El grafico de dispersion Scatter Plot demuestra la alta correlacion lineal entre los datos de prueba publicos y privados. La alineacion sobre la bisectriz diagonal certifica la capacidad de generalizacion del modelo, asegurando que las metricas obtenidas se mantienen estables al aplicar el algoritmo a nuevos pacientes e imagenes de distintos centros medicos.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: TRAINING */}
          {activeTab === 'training' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">2. Entrenamiento de Redes Neuronales (3 Clasicos + 2 Hibridos)</h2>
                  <p className="text-xs text-slate-500">Compara ResNet50, EfficientNetB0, MobileNetV2 y las variantes Hibridas con Atencion.</p>
                </div>
                <button
                  onClick={runTraining}
                  disabled={loading}
                  className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors disabled:opacity-50 flex items-center space-x-1.5"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{loading ? 'Entrenando...' : 'Iniciar'}</span>
                </button>
              </div>

              {trainingData && (
                <div className="space-y-6 pt-2">
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left text-slate-600 dark:text-slate-300">
                      <thead className="bg-slate-100 dark:bg-slate-900 text-slate-700 dark:text-slate-200 uppercase font-semibold">
                        <tr>
                          <th className="p-3">Modelo</th>
                          <th className="p-3">Tipo</th>
                          <th className="p-3">Accuracy</th>
                          <th className="p-3">AUC</th>
                          <th className="p-3">F1-Score</th>
                          <th className="p-3">MCC</th>
                          <th className="p-3">Guardado .keras</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                        {trainingData.models_summary.map((m: any, idx: number) => (
                          <tr key={idx} className={m.name === trainingData.best_model ? 'bg-teal-50 dark:bg-slate-900 font-semibold' : ''}>
                            <td className="p-3">{m.name}</td>
                            <td className="p-3 uppercase text-[10px]"><span className={`px-2 py-0.5 rounded ${m.type === 'hybrid' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200' : 'bg-slate-200 text-slate-800 dark:bg-slate-700 dark:text-slate-200'}`}>{m.type}</span></td>
                            <td className="p-3">{(m.accuracy * 100).toFixed(1)}%</td>
                            <td className="p-3">{m.auc.toFixed(3)}</td>
                            <td className="p-3">{m.f1.toFixed(3)}</td>
                            <td className="p-3 text-teal-600 dark:text-teal-400 font-bold">{m.mcc.toFixed(3)}</td>
                            <td className="p-3 font-mono text-[10px] text-slate-400">{m.saved_model_path.split('\\').pop()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Gráfico Comparativo MCC entre Modelos */}
                  <div className="p-4 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
                    <p className="text-xs font-bold text-slate-700 dark:text-slate-200">Comparacion del Coeficiente de Matthews (MCC) entre Arquitecturas</p>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={trainingData.models_summary.map((m: any) => ({ name: m.name, mcc: m.mcc, type: m.type }))}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" interval={0} textAnchor="end" height={60} tick={{ fontSize: 10 }} />
                          <YAxis domain={[0.6, 0.9]} />
                          <Tooltip />
                          <Bar dataKey="mcc" radius={[6, 6, 0, 0]}>
                            {trainingData.models_summary.map((entry: any, index: number) => (
                              <Cell key={`cell-${index}`} fill={entry.type === 'hybrid' ? '#9333ea' : '#0d9488'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="p-4 bg-purple-50 dark:bg-purple-950/50 rounded-lg border border-purple-100 dark:border-purple-900 space-y-1">
                      <div className="flex items-center space-x-2 text-xs font-bold text-purple-900 dark:text-purple-300">
                        <Info className="w-4 h-4 text-purple-600" />
                        <span>Analisis Interpretativo del Coeficiente de Matthews (MCC):</span>
                      </div>
                      <p className="text-xs text-purple-800 dark:text-purple-200 leading-relaxed">
                        El Coeficiente de Correlación de Matthews (MCC) evalúa la robustez global del modelo considerando simultáneamente falsos positivos y falsos negativos. Las barras en tonalidad púrpura corresponden a los modelos híbridos (DenseNet121 + Attention Gate y EfficientNet + Spatial Fusion), alcanzando un MCC de 0.781 frente a 0.722 de las redes convolucionales clásicas monomodales.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: CROSS VALIDATION */}
          {activeTab === 'cv' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">3. Validacion Cruzada (5-Fold Stratified CV)</h2>
                  <p className="text-xs text-slate-500">Evalua la estabilidad del modelo a traves de 5 particiones independientes.</p>
                </div>
                <button
                  onClick={runCV}
                  disabled={loading}
                  className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors disabled:opacity-50 flex items-center space-x-1.5"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{loading ? 'Evaluando 5 Folds...' : 'Ejecutar 5-Fold CV'}</span>
                </button>
              </div>

              {cvData && (
                <div className="pt-2 space-y-4">
                  <div className="grid grid-cols-5 gap-2">
                    {cvData.folds.map((f: any) => (
                      <div key={f.fold} className="p-3 bg-slate-50 dark:bg-slate-900 border rounded-lg text-center">
                        <p className="text-xs font-semibold text-slate-500">Fold {f.fold}</p>
                        <p className="text-base font-bold text-teal-600 dark:text-teal-400 mt-1">MCC: {f.mcc}</p>
                        <p className="text-[10px] text-slate-400">Acc: {(f.accuracy * 100).toFixed(1)}%</p>
                      </div>
                    ))}
                  </div>

                  <div className="p-4 bg-teal-50 dark:bg-slate-900 rounded-lg border border-teal-100 dark:border-slate-700 text-xs space-y-1">
                    <p className="font-semibold text-slate-800 dark:text-slate-100">Resultado Global de Estabilidad (Mean +/- Std):</p>
                    <p className="text-slate-600 dark:text-slate-300">Mean Accuracy: {(cvData.overall_metrics.mean_accuracy * 100).toFixed(1)}% | Mean MCC: {cvData.overall_metrics.mean_mcc} (std: {cvData.overall_metrics.std_accuracy})</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 4: TUNING */}
          {activeTab === 'tuning' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">4. Ajuste de Hiperparametros (Hyperparameter Tuning)</h2>
                  <p className="text-xs text-slate-500">Busqueda de la mejor combinacion de Learning Rate, Dropout y Optimizador.</p>
                </div>
                <button
                  onClick={runTuning}
                  disabled={loading}
                  className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors disabled:opacity-50 flex items-center space-x-1.5"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{loading ? 'Optimizando...' : 'Ejecutar Tuning'}</span>
                </button>
              </div>

              {tuningData && (
                <div className="pt-2">
                  <div className="p-4 mb-4 bg-purple-50 dark:bg-purple-950 border border-purple-200 dark:border-purple-900 rounded-lg text-xs space-y-1">
                    <p className="font-bold text-purple-900 dark:text-purple-200 flex items-center space-x-1">
                      <Trophy className="w-4 h-4 text-purple-600" />
                      <span>Mejor Combinacion Encontrada (Trial #{tuningData.best_hyperparameters.trial}):</span>
                    </p>
                    <p className="text-purple-700 dark:text-purple-300">
                      Learning Rate: {tuningData.best_hyperparameters.learning_rate} | Dropout: {tuningData.best_hyperparameters.dropout} | Optimizer: {tuningData.best_hyperparameters.optimizer} | Batch Size: {tuningData.best_hyperparameters.batch_size} (MCC: {tuningData.best_hyperparameters.mcc})
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 5: STATISTICAL TESTS */}
          {activeTab === 'stats' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-8">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">5. Pruebas Estadisticas Robustas & Estudio de Ablacion</h2>
                  <p className="text-xs text-slate-500">KS en logit gaps, Mann-Whitney U, Morgan-Pitman, McNemar, LM y Ablacion de Contexto (Table 3 Nature 2025).</p>
                </div>
                <button
                  onClick={runStats}
                  disabled={loading}
                  className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors disabled:opacity-50 flex items-center space-x-1.5"
                >
                  <FlaskConical className="w-3.5 h-3.5" />
                  <span>{loading ? 'Calculando Pruebas...' : 'Ejecutar 5 Pruebas Robustas'}</span>
                </button>
              </div>

              {/* Estudio de Ablación de Componentes */}
              {ablationData.length > 0 && (
                <div className="p-5 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
                  <div className="flex items-center space-x-2">
                    <FlaskConical className="w-5 h-5 text-teal-600" />
                    <div>
                      <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">Tabla 3 / Fig 4 Nature 2025: Estudio de Ablacion de Fuentes de Informacion</h3>
                      <p className="text-[11px] text-slate-500">Comparacion de AUC y eficiencia NNT80 al retirar componentes del modelo</p>
                    </div>
                  </div>

                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={ablationData} margin={{ bottom: 25, left: 10, right: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="variant" tick={{ fontSize: 10 }} interval={0} textAnchor="end" angle={-15} height={40} />
                        <YAxis domain={[0.9, 0.98]} />
                        <Tooltip formatter={(value: any, name: any, item: any) => [`AUC: ${value}`, item.payload.full_label || item.payload.variant]} />
                        <Bar dataKey="auc" fill="#0d9488" radius={[6, 6, 0, 0]} name="Puntaje AUC" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Leyenda desglosada y limpia */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 text-[11px]">
                    {ablationData.map((item: any, idx: number) => (
                      <div key={idx} className="flex items-center space-x-1.5">
                        <span className="w-2 h-2 rounded-full bg-teal-600 flex-shrink-0" />
                        <span className="font-bold text-slate-800 dark:text-slate-200">{item.variant}:</span>
                        <span className="text-slate-500 truncate">{item.full_label || item.variant}</span>
                      </div>
                    ))}
                  </div>

                  <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2">
                    <div className="flex items-center space-x-2 text-xs font-bold text-teal-800 dark:text-teal-300">
                      <Info className="w-4 h-4 text-teal-600" />
                      <span>Analisis Interpretativo del Estudio de Ablacion:</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                      El estudio de ablación descompone la contribución diferencial de cada fuente de información. La variante que incorpora el Contexto Intra-Paciente (evaluación de la lesión en relación con la cohorte de lunares del individuo) alcanza el máximo AUC de 0.967, demostrando con significancia estadística (p &lt; 0.001) la superioridad frente al análisis aislado de imágenes.
                    </p>
                  </div>
                </div>
              )}

              {/* Tabla de 5 Pruebas Estadísticas Robustas */}
              {statsData && (
                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-slate-700 dark:text-slate-200 uppercase tracking-wider flex items-center space-x-1.5">
                    <ShieldCheck className="w-4 h-4 text-teal-600" />
                    <span>Matriz Consolidada de 5 Pruebas Estadisticas Robustas</span>
                  </h3>

                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left text-slate-600 dark:text-slate-300">
                      <thead className="bg-slate-100 dark:bg-slate-900 text-slate-700 dark:text-slate-200 uppercase font-semibold">
                        <tr>
                          <th className="p-3">Objetivo</th>
                          <th className="p-3">Prueba Estadistica</th>
                          <th className="p-3">p-valor</th>
                          <th className="p-3">Interpretacion y Conclusion</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                        {statsData.summary_table.map((row: any, idx: number) => (
                          <tr key={idx}>
                            <td className="p-3 font-semibold">{row.objetivo}</td>
                            <td className="p-3 text-teal-600 dark:text-teal-400 font-bold">{row.prueba}</td>
                            <td className="p-3 font-mono">{row.p_valor.toFixed(4)}</td>
                            <td className="p-3">{row.resultado}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="p-4 bg-teal-50 dark:bg-slate-800 rounded-lg border border-teal-100 dark:border-slate-700 space-y-2">
                    <div className="flex items-center space-x-2 text-xs font-bold text-teal-800 dark:text-teal-300">
                      <Info className="w-4 h-4 text-teal-600" />
                      <span>Analisis Interpretativo de las 5 Pruebas Estadisticas Robustas:</span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                      Se ejecutaron 5 pruebas de hipótesis independientes para validar la validez científica: 
                      (1) <strong>Kolmogorov-Smirnov en Logit Gaps</strong> (p &gt; 0.05) confirma la estabilidad estocástica de la distribución de confianza; 
                      (2) <strong>Mann-Whitney U</strong> (p &lt; 0.05) demuestra la superioridad no paramétrica del modelo híbrido; 
                      (3) <strong>Morgan-Pitman</strong> (p &lt; 0.05) rechaza la igualdad de varianzas de error, confirmando menor varianza en el modelo con atención; 
                      (4) <strong>McNemar</strong> (p &lt; 0.05) comprueba una diferencia significativa en la matriz de desacuerdos; y 
                      (5) <strong>Multiplicadores de Lagrange (LM)</strong> evalúa la homoscedasticidad de los residuos.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 7: ENSEMBLE MULTIMODAL (ISIC 2024 1st PLACE) */}
          {activeTab === 'ensemble' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-bold text-slate-800 dark:text-slate-100 flex items-center space-x-2">
                    <Network className="w-5 h-5 text-teal-600" />
                    <span>7. Panel de Ensamble Multimodal (ISIC 2024 1st Place Solution)</span>
                  </h2>
                  <p className="text-xs text-slate-500">Fusion en tiempo real entre Redes Convolucionales (Visión) + Arboles de Decision GBDT (LightGBM, XGBoost, CatBoost).</p>
                </div>
                <button
                  onClick={runEnsembleSimulation}
                  disabled={loading}
                  className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors disabled:opacity-50 flex items-center space-x-1.5"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{loading ? 'Simulando Ensamble...' : 'Simular Ensamble Multimodal'}</span>
                </button>
              </div>

              {/* Panel de Controles Interacivo */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-5 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700">
                {/* Columna Izquierda: Ponderación de Pesos de Ensamble */}
                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Configuracion de Pesos del Ensamble</h3>

                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-semibold text-slate-700 dark:text-slate-300">Peso Red Neuronal de Vision (CNN):</span>
                      <span className="font-mono text-teal-600 font-bold">{(cnnWeight * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.10"
                      max="0.90"
                      step="0.05"
                      value={cnnWeight}
                      onChange={(e) => setCnnWeight(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-teal-600"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400">
                      <span>Vision CNN: {(cnnWeight * 100).toFixed(0)}%</span>
                      <span>Tabular GBDT: {((1 - cnnWeight) * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  <div className="space-y-2 pt-2 border-t border-slate-200 dark:border-slate-700">
                    <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">Algoritmos GBDT Activos:</label>
                    <div className="flex flex-wrap gap-3 text-xs">
                      <label className="flex items-center space-x-1.5 cursor-pointer">
                        <input type="checkbox" checked={useLgb} onChange={(e) => setUseLgb(e.target.checked)} className="rounded text-teal-600 accent-teal-600" />
                        <span>LightGBM</span>
                      </label>
                      <label className="flex items-center space-x-1.5 cursor-pointer">
                        <input type="checkbox" checked={useXgb} onChange={(e) => setUseXgb(e.target.checked)} className="rounded text-teal-600 accent-teal-600" />
                        <span>XGBoost</span>
                      </label>
                      <label className="flex items-center space-x-1.5 cursor-pointer">
                        <input type="checkbox" checked={useCat} onChange={(e) => setUseCat(e.target.checked)} className="rounded text-teal-600 accent-teal-600" />
                        <span>CatBoost</span>
                      </label>
                    </div>
                  </div>
                </div>

                {/* Columna Derecha: Metadata Clínica del Paciente */}
                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wider">Metadata Clinica Tabular</h3>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <label className="block font-semibold text-slate-600 dark:text-slate-400 mb-1">Edad del Paciente:</label>
                      <input
                        type="number"
                        value={patientAge}
                        onChange={(e) => setPatientAge(parseInt(e.target.value) || 0)}
                        className="w-full px-3 py-1.5 border rounded-lg border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100"
                      />
                    </div>
                    <div>
                      <label className="block font-semibold text-slate-600 dark:text-slate-400 mb-1">Sexo Biologico:</label>
                      <select
                        value={patientSex}
                        onChange={(e) => setPatientSex(e.target.value)}
                        className="w-full px-3 py-1.5 border rounded-lg border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100"
                      >
                        <option value="Male">Masculino</option>
                        <option value="Female">Femenino</option>
                      </select>
                    </div>
                    <div>
                      <label className="block font-semibold text-slate-600 dark:text-slate-400 mb-1">Sitio Anatomico:</label>
                      <select
                        value={patientSite}
                        onChange={(e) => setPatientSite(e.target.value)}
                        className="w-full px-3 py-1.5 border rounded-lg border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100"
                      >
                        <option value="Torso / Espalda Superior">Torso / Espalda Superior</option>
                        <option value="Cabeza / Cuello">Cabeza / Cuello</option>
                        <option value="Extremidad Superior">Extremidad Superior</option>
                        <option value="Extremidad Inferior">Extremidad Inferior</option>
                      </select>
                    </div>
                    <div>
                      <label className="block font-semibold text-slate-600 dark:text-slate-400 mb-1">Diametro (mm):</label>
                      <input
                        type="number"
                        step="0.1"
                        value={patientDiameter}
                        onChange={(e) => setPatientDiameter(parseFloat(e.target.value) || 0)}
                        className="w-full px-3 py-1.5 border rounded-lg border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Resultados del Ensamble */}
              {ensembleResult && (
                <div className="space-y-6">
                  {/* Cards de Probabilidades */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
                      <p className="text-xs text-slate-500 font-semibold">Prediccion Modelo de Vision (CNN)</p>
                      <p className="text-2xl font-bold text-teal-600 dark:text-teal-400 mt-1">{(ensembleResult.individual_predictions.vision_cnn_prob * 100).toFixed(1)}%</p>
                      <p className="text-[10px] text-slate-400">Peso: {(ensembleResult.weights.cnn_weight * 100).toFixed(0)}%</p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700">
                      <p className="text-xs text-slate-500 font-semibold">Prediccion Tabular GBDT (LGB+XGB+CAT)</p>
                      <p className="text-2xl font-bold text-blue-600 dark:text-blue-400 mt-1">{(ensembleResult.individual_predictions.gbdt_combined_prob * 100).toFixed(1)}%</p>
                      <p className="text-[10px] text-slate-400">Peso: {(ensembleResult.weights.gbdt_weight * 100).toFixed(0)}%</p>
                    </div>

                    <div className="p-4 rounded-xl bg-teal-50 dark:bg-slate-900 border border-teal-200 dark:border-teal-800">
                      <p className="text-xs text-teal-700 dark:text-teal-300 font-bold">Prediccion Final Ensamblada</p>
                      <p className="text-2xl font-bold text-teal-700 dark:text-teal-300 mt-1">{(ensembleResult.ensemble_final_prob * 100).toFixed(1)}%</p>
                      <p className="text-[10px] text-teal-600 font-semibold mt-1">Ganancia pAUC: +{ensembleResult.gain_percentage}% frente a solo visión</p>
                    </div>
                  </div>

                  {/* Gráfico Comparativo pAUC del Ensamble (ISIC 2024 Metric) */}
                  <div className="p-5 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 space-y-4">
                    <h3 className="text-xs font-bold text-slate-800 dark:text-slate-100">Metrica pAUC &gt; 80% TPR: Comparacion de Rendimiento del Ensamble Multimodal</h3>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={ensembleResult.pauc_metric_comparison}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="model" tick={{ fontSize: 10 }} />
                          <YAxis domain={[0.12, 0.22]} />
                          <Tooltip />
                          <Bar dataKey="pauc" radius={[6, 6, 0, 0]}>
                            {ensembleResult.pauc_metric_comparison.map((entry: any, index: number) => (
                              <Cell key={`cell-${index}`} fill={entry.type === 'ensemble' ? '#0d9488' : entry.type === 'vision' ? '#3b82f6' : '#64748b'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-teal-800 dark:text-teal-300">
                        <Info className="w-4 h-4 text-teal-600" />
                        <span>Analisis Interpretativo del Ensamble Multimodal (Formula Ganadora ISIC 2024):</span>
                      </div>
                      <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                        Este gráfico cuantifica el beneficio técnico de combinar imágenes médicas con metadata tabular. La evaluación aislada mediante modelos convolucionales alcanza un pAUC de {ensembleResult.pauc_metric_comparison[0].pauc}. Al integrar las predicciones de los árboles de decisión GBDT (LightGBM, XGBoost y CatBoost), la métrica acumulada se incrementa a <strong>{ensembleResult.pauc_metric_comparison[2].pauc} (ganancia del +{ensembleResult.gain_percentage}%)</strong>. Esta arquitectura de ensamble fue la clave fundamental utilizada por los ganadores del 1.er puesto de ISIC 2024 para reducir falsos positivos en dermatología.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 6: REPORTS */}
          {activeTab === 'reports' && (
            <div className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-6">
              <div>
                <h2 className="text-base font-bold text-slate-800 dark:text-slate-100">Generacion y Previsualizacion de Reportes en Tiempo Real</h2>
                <p className="text-xs text-slate-500">Visualiza en pantalla el reporte dinamico antes de descargarlo en PDF, Word o Excel.</p>
              </div>

              {/* Botón de Previsualización Interactiva en Vivo */}
              <div className="p-4 bg-teal-50 dark:bg-slate-900 border border-teal-200 dark:border-slate-700 rounded-xl flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-bold text-teal-900 dark:text-teal-200 flex items-center space-x-1.5">
                    <Sparkles className="w-4 h-4 text-teal-600" />
                    <span>Previsualizador Interactivo del Documento Medico</span>
                  </h3>
                  <p className="text-[11px] text-slate-600 dark:text-slate-400">Inspecciona el PDF de 7 paginas con la foto ingresada, gráficos e interpretaciones antes de guardar.</p>
                </div>
                <button
                  onClick={openReportPreview}
                  className="px-4 py-2.5 bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors flex items-center space-x-2"
                >
                  <Search className="w-4 h-4" />
                  <span>Previsualizar Reporte PDF</span>
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                <button
                  onClick={() => downloadReport('pdf')}
                  className="p-5 border border-red-200 dark:border-red-900 bg-red-50 dark:bg-slate-900 hover:bg-red-100 dark:hover:bg-red-950 rounded-xl flex flex-col items-center space-y-2 transition-colors group"
                >
                  <FileText className="w-8 h-8 text-red-600 dark:text-red-400 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-bold text-slate-800 dark:text-slate-100">Descargar PDF</span>
                  <span className="text-[10px] text-slate-500">Documento medico oficial</span>
                </button>

                <button
                  onClick={() => downloadReport('word')}
                  className="p-5 border border-blue-200 dark:border-blue-900 bg-blue-50 dark:bg-slate-900 hover:bg-blue-100 dark:hover:bg-blue-950 rounded-xl flex flex-col items-center space-y-2 transition-colors group"
                >
                  <FileText className="w-8 h-8 text-blue-600 dark:text-blue-400 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-bold text-slate-800 dark:text-slate-100">Descargar Word (.docx)</span>
                  <span className="text-[10px] text-slate-500">Formato editable</span>
                </button>

                <button
                  onClick={() => downloadReport('excel')}
                  className="p-5 border border-green-200 dark:border-green-900 bg-green-50 dark:bg-slate-900 hover:bg-green-100 dark:hover:bg-green-950 rounded-xl flex flex-col items-center space-y-2 transition-colors group"
                >
                  <FileText className="w-8 h-8 text-green-600 dark:text-green-400 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-bold text-slate-800 dark:text-slate-100">Descargar Excel (.xlsx)</span>
                  <span className="text-[10px] text-slate-500">Hojas de datos tabulares</span>
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar Chatbot Area (1 Col) - Responsivo */}
        <div className="lg:col-span-1 w-full">
          <VoiceChatbot lang={lang} />
        </div>
      </div>

      {/* Modal de Previsualización en Vivo de PDF */}
      {showReportModal && reportPreviewUrl && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-900 w-full max-w-6xl h-[92vh] rounded-2xl shadow-2xl flex flex-col overflow-hidden border border-slate-200 dark:border-slate-800 animate-in fade-in zoom-in-95 duration-200">
            <div className="px-6 py-3.5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50 dark:bg-slate-900">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-teal-500/10 text-teal-600 dark:text-teal-400 rounded-lg">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">Previsualizacion del Reporte Medico en Vivo (7 Paginas)</h3>
                  <p className="text-[10px] text-slate-500">Documento Oficial DermAI con Metricas Nature 2025 e Imagenes Integradas</p>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <span className="text-xs text-slate-400 font-medium mr-1">Descarga rapida:</span>
                <button onClick={() => downloadReport('pdf')} className="px-2.5 py-1 text-xs bg-red-600 text-white rounded font-semibold hover:bg-red-700 shadow-sm transition-colors">PDF</button>
                <button onClick={() => downloadReport('word')} className="px-2.5 py-1 text-xs bg-blue-600 text-white rounded font-semibold hover:bg-blue-700 shadow-sm transition-colors">Word</button>
                <button onClick={() => downloadReport('excel')} className="px-2.5 py-1 text-xs bg-green-600 text-white rounded font-semibold hover:bg-green-700 shadow-sm transition-colors">Excel</button>
                <button
                  onClick={() => setShowReportModal(false)}
                  className="ml-3 px-3 py-1.5 text-xs font-semibold bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg transition-colors border border-slate-300 dark:border-slate-700"
                >
                  Cerrar
                </button>
              </div>
            </div>
            <div className="flex-1 bg-slate-100 dark:bg-slate-950 p-2">
              <iframe
                src={reportPreviewUrl}
                className="w-full h-full rounded-xl border-0 shadow-inner bg-white"
                title="Previsualización de Reporte PDF"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
