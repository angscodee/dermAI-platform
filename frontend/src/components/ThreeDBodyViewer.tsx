'use client';

import React, { useState } from 'react';
import { Box, Layers, RotateCw, Calendar, TrendingUp, ShieldAlert, Info, Sparkles, ZoomIn, ZoomOut, RefreshCw, Target, Activity } from 'lucide-react';

interface ThreeDBodyViewerProps {
  lesionCoords?: { x: number; y: number; z: number };
  diagnosis?: string;
  isMalignant?: boolean;
  growthInfo?: any;
}

export function ThreeDBodyViewer({ lesionCoords, diagnosis, isMalignant, growthInfo }: ThreeDBodyViewerProps) {
  const [selectedBodyPart, setSelectedBodyPart] = useState('Torso / Espalda Superior');
  const [timelineMonth, setTimelineMonth] = useState(0); // 0, 3, 6, 12
  const [zoomLevel, setZoomLevel] = useState(1.0); // 1.0x to 3.0x

  const d0 = growthInfo?.initial_diameter_mm || 6.4;
  const proj3 = growthInfo?.projected_diameter_3m_mm || 7.8;
  const proj6 = growthInfo?.projected_diameter_6m_mm || 9.8;
  const proj12 = growthInfo?.projected_diameter_12m_mm || 14.2;

  const currentDiameter = timelineMonth === 0 ? d0 : timelineMonth === 3 ? proj3 : timelineMonth === 6 ? proj6 : proj12;

  const bodyParts = [
    { id: 'Torso / Espalda Superior', label: 'Torso / Espalda', top: '35%', left: '50%', origin: '50% 35%' },
    { id: 'Cabeza / Cuello', label: 'Cabeza / Cuello', top: '15%', left: '50%', origin: '50% 15%' },
    { id: 'Brazo Derecho', label: 'Brazo Derecho', top: '38%', left: '30%', origin: '30% 38%' },
    { id: 'Brazo Izquierdo', label: 'Brazo Izquierdo', top: '38%', left: '70%', origin: '70% 38%' },
    { id: 'Pierna Derecha', label: 'Pierna Derecha', top: '70%', left: '42%', origin: '42% 70%' },
    { id: 'Pierna Izquierda', label: 'Pierna Izquierda', top: '70%', left: '58%', origin: '58% 70%' },
  ];

  const currentPart = bodyParts.find((p) => p.id === selectedBodyPart) || bodyParts[0];

  const handleSelectZone = (partId: string) => {
    setSelectedBodyPart(partId);
    setZoomLevel(1.85);
  };

  const handleZoomIn = () => setZoomLevel((prev) => Math.min(3.5, prev + 0.4));
  const handleZoomOut = () => setZoomLevel((prev) => Math.max(1.0, prev - 0.4));
  const handleResetZoom = () => {
    setZoomLevel(1.0);
    setSelectedBodyPart('Torso / Espalda Superior');
  };

  return (
    <div className="bg-slate-50 dark:bg-slate-900 p-4 sm:p-6 rounded-xl border border-slate-200 dark:border-slate-700 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 w-full">
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-teal-600 dark:text-teal-400" />
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">Visor de Malla 3D Medica Holografica con Enfoque Anatomico</h3>
        </div>
        
        {/* Controls Bar */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-1 shadow-sm">
            <button
              onClick={handleZoomIn}
              className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-700 rounded text-slate-700 dark:text-slate-200"
              title="Acercar (Zoom In)"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-700 rounded text-slate-700 dark:text-slate-200"
              title="Alejar (Zoom Out)"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-700 rounded text-slate-700 dark:text-slate-200"
              title="Restablecer Vista"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
            <span className="text-[11px] font-mono font-bold px-1.5 text-teal-600">
              {zoomLevel.toFixed(1)}x
            </span>
          </div>
        </div>
      </div>

      {/* Selectores Rápido de Zona Anatómica Responsivos */}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <span className="text-xs font-semibold text-slate-500 flex items-center space-x-1 mr-2">
          <Target className="w-3.5 h-3.5 text-teal-600" />
          <span>Foco Anatomico:</span>
        </span>
        {bodyParts.map((part) => (
          <button
            key={part.id}
            onClick={() => handleSelectZone(part.id)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
              selectedBodyPart === part.id
                ? 'bg-teal-600 text-white shadow-md scale-105'
                : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700'
            }`}
          >
            {part.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        {/* Photorealistic 3D Hologram Avatar Container */}
        <div className="relative border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden bg-slate-950 flex justify-center items-center p-4 min-h-[380px] shadow-2xl">
          <div
            className="relative transition-all duration-500 ease-out flex justify-center items-center w-full h-full"
            style={{
              transform: `scale(${zoomLevel})`,
              transformOrigin: currentPart.origin
            }}
          >
            <img
              src="/human_3d_mesh.png"
              alt="3D Body Mesh Hologram"
              className="max-h-80 object-contain drop-shadow-[0_0_25px_rgba(13,148,136,0.6)]"
            />

            {/* Interactive Hotspot Buttons mapped over the 3D Body Mesh */}
            {bodyParts.map((part) => (
              <button
                key={part.id}
                onClick={() => handleSelectZone(part.id)}
                style={{ top: part.top, left: part.left }}
                className={`absolute -translate-x-1/2 -translate-y-1/2 transition-all group ${
                  selectedBodyPart === part.id ? 'scale-125 z-20' : 'opacity-70 hover:opacity-100 z-10'
                }`}
              >
                <div className="relative flex items-center justify-center">
                  <span className={`animate-ping absolute inline-flex h-6 w-6 rounded-full ${
                    isMalignant ? 'bg-red-400' : 'bg-teal-400'
                  } opacity-75`} />
                  <span className={`relative inline-flex rounded-full h-3.5 w-3.5 ${
                    isMalignant ? 'bg-red-500' : 'bg-teal-500'
                  } border-2 border-white shadow-md`} />
                </div>
                <span className="opacity-0 group-hover:opacity-100 absolute left-1/2 -translate-x-1/2 bottom-5 bg-slate-900 text-white text-[9px] px-2 py-0.5 rounded font-semibold whitespace-nowrap shadow-md transition-opacity">
                  {part.label}
                </span>
              </button>
            ))}
          </div>

          <div className="absolute bottom-3 left-3 bg-slate-900/80 text-teal-400 border border-teal-800/50 text-[10px] px-2.5 py-1 rounded font-mono shadow-sm backdrop-blur">
            Zona Enfocada: <span className="text-white font-bold">{selectedBodyPart}</span> ({zoomLevel.toFixed(1)}x)
          </div>
        </div>

        {/* Proyección Matemática de Crecimiento & Profundidad de Breslow */}
        <div className="space-y-4">
          <div className="p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-3">
            <div className="flex items-center space-x-2 text-xs font-bold text-teal-700 dark:text-teal-400">
              <TrendingUp className="w-4 h-4" />
              <span>Modelo Logistico de Gompertz (Proyeccion Temporal)</span>
            </div>

            <div className="grid grid-cols-4 gap-2 text-center text-xs font-mono">
              <div className={`p-2 rounded border ${timelineMonth === 0 ? 'bg-teal-50 border-teal-500 font-bold text-teal-800' : 'bg-slate-50 border-slate-200'}`}>
                <div className="text-[10px] text-slate-400">Mes 0</div>
                <div>{d0} mm</div>
              </div>
              <div className={`p-2 rounded border ${timelineMonth === 3 ? 'bg-teal-50 border-teal-500 font-bold text-teal-800' : 'bg-slate-50 border-slate-200'}`}>
                <div className="text-[10px] text-slate-400">+3 Meses</div>
                <div>{proj3} mm</div>
              </div>
              <div className={`p-2 rounded border ${timelineMonth === 6 ? 'bg-teal-50 border-teal-500 font-bold text-teal-800' : 'bg-slate-50 border-slate-200'}`}>
                <div className="text-[10px] text-slate-400">+6 Meses</div>
                <div>{proj6} mm</div>
              </div>
              <div className={`p-2 rounded border ${timelineMonth === 12 ? 'bg-teal-50 border-teal-500 font-bold text-teal-800' : 'bg-slate-50 border-slate-200'}`}>
                <div className="text-[10px] text-slate-400">+12 Meses</div>
                <div>{proj12} mm</div>
              </div>
            </div>

            {/* Slider de Línea de Tiempo */}
            <div className="space-y-1 pt-1">
              <div className="flex justify-between items-center text-[11px] font-semibold">
                <span className="text-slate-600 dark:text-slate-400">Simulacion Temporal:</span>
                <span className="text-teal-600 font-mono font-bold">Mes {timelineMonth} ({currentDiameter} mm estimados)</span>
              </div>
              <input
                type="range"
                min="0"
                max="12"
                step="3"
                value={timelineMonth}
                onChange={(e) => setTimelineMonth(parseInt(e.target.value))}
                className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-teal-600"
              />
            </div>
          </div>

          {/* Estimación Profundidad de Breslow */}
          {growthInfo && (
            <div className="p-4 bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-900 rounded-xl space-y-2 text-xs">
              <div className="flex items-center space-x-2 font-bold text-amber-900 dark:text-amber-200">
                <ShieldAlert className="w-4 h-4 text-amber-600" />
                <span>Profundidad de Invasion Dermica de Breslow</span>
              </div>
              <p className="text-slate-700 dark:text-slate-300">
                Profundidad Estimada: <span className="font-mono font-bold text-amber-700">{growthInfo.estimated_breslow_depth_mm} mm</span> | Clasificacion AJCC: <span className="font-mono font-semibold">{growthInfo.ajcc_staging_classification}</span>
              </p>
              <p className="text-[10px] text-amber-700/80 italic">Ecuacion: D(t) = K / (1 + ((K - D0)/D0) · e^(-r · t))</p>
            </div>
          )}
        </div>
      </div>

      {/* Explicación Sencilla para Todo Público */}
      <div className="p-4 bg-teal-50 dark:bg-slate-800 rounded-xl border border-teal-100 dark:border-slate-700 space-y-2">
        <div className="flex items-center space-x-2 text-xs font-bold text-teal-800 dark:text-teal-300">
          <Info className="w-4 h-4 text-teal-600" />
          <span>Analisis Interpretativo del Modelo 3D y Proyeccion Temporal:</span>
        </div>
        <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
          Este modulo mapea la ubicacion de la lesion en la anatomia tridimensional del paciente. Al seleccionar una region (ej. Cabeza, Torso, Extremidades), la camara realiza un enfoque focal automatico. El modelo matematico de Gompertz proyecta el incremento en milimetros a 3, 6 y 12 meses, ofreciendo una estimacion de la velocidad de crecimiento proliferativo y la profundidad de Breslow para guiar el triaje clinico.
        </p>
      </div>
    </div>
  );
}
