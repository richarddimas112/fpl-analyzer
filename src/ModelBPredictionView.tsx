import React, { useState, useMemo } from 'react';
import {
  SlidersHorizontal,
  RotateCcw,
  Sparkles,
  Shield,
  TrendingUp,
  Award,
  Search,
  ChevronDown,
  Info,
  Layers,
  ArrowUpDown,
  BarChart3,
  Target
} from 'lucide-react';

export interface PlayerOptionB {
  id: number;
  Nama_Pemain: string;
  Klub: string;
  Posisi: 'FWD' | 'MID' | 'DEF' | 'GK';
  Harga: number;
  Lawan_GW_Berikutnya: string;
  Status: string;
  Peluang_CS: number;
  Diff_Attack_Team: number;
  Diff_Defense_Team: number;
  xMins_Pts: number;
  xG_Pred: number;
  xA_Pred: number;
  xSaves_Pts: number;
  xDC_Pts: number;
  xCS_Pts: number;
  xBP: number;
  Avg_Mins_L5M: number;
}

export interface PositionMultipliers {
  FWD: { xG: number; xA: number };
  MID: { xG: number; xA: number };
  DEF: { xG: number; xA: number };
  GK: { xG: number; xA: number };
}

export const DEFAULT_MULTIPLIERS: PositionMultipliers = {
  FWD: { xG: 1.08, xA: 1.05 },
  MID: { xG: 1.04, xA: 1.06 },
  DEF: { xG: 0.88, xA: 0.92 },
  GK: { xG: 0.00, xA: 0.50 }
};

const GOAL_POINTS_MAP = {
  GK: 10.0,
  DEF: 6.0,
  MID: 5.0,
  FWD: 4.0
};

// Initial benchmark players representative of FPL 2026 dataset
export const BENCHMARK_PLAYERS: PlayerOptionB[] = [
  {
    id: 1,
    Nama_Pemain: 'Erling Haaland',
    Klub: 'Man City',
    Posisi: 'FWD',
    Harga: 15.2,
    Lawan_GW_Berikutnya: 'BUR (H)',
    Status: 'Tersedia',
    Peluang_CS: 44.5,
    Diff_Attack_Team: 18.2,
    Diff_Defense_Team: 16.5,
    xMins_Pts: 1.95,
    xG_Pred: 0.92,
    xA_Pred: 0.22,
    xSaves_Pts: 0.0,
    xDC_Pts: 0.15,
    xCS_Pts: 0.0,
    xBP: 1.45,
    Avg_Mins_L5M: 88.0
  },
  {
    id: 2,
    Nama_Pemain: 'Mohamed Salah',
    Klub: 'Liverpool',
    Posisi: 'MID',
    Harga: 13.0,
    Lawan_GW_Berikutnya: 'EVE (H)',
    Status: 'Tersedia',
    Peluang_CS: 42.0,
    Diff_Attack_Team: 21.0,
    Diff_Defense_Team: 12.8,
    xMins_Pts: 1.95,
    xG_Pred: 0.68,
    xA_Pred: 0.44,
    xSaves_Pts: 0.0,
    xDC_Pts: 0.35,
    xCS_Pts: 0.42,
    xBP: 1.25,
    Avg_Mins_L5M: 89.0
  },
  {
    id: 3,
    Nama_Pemain: 'Bukayo Saka',
    Klub: 'Arsenal',
    Posisi: 'MID',
    Harga: 10.2,
    Lawan_GW_Berikutnya: 'SOU (H)',
    Status: 'Tersedia',
    Peluang_CS: 51.5,
    Diff_Attack_Team: 24.5,
    Diff_Defense_Team: 22.0,
    xMins_Pts: 1.90,
    xG_Pred: 0.55,
    xA_Pred: 0.52,
    xSaves_Pts: 0.0,
    xDC_Pts: 0.40,
    xCS_Pts: 0.51,
    xBP: 1.15,
    Avg_Mins_L5M: 86.5
  },
  {
    id: 4,
    Nama_Pemain: 'Cole Palmer',
    Klub: 'Chelsea',
    Posisi: 'MID',
    Harga: 10.8,
    Lawan_GW_Berikutnya: 'NFO (H)',
    Status: 'Tersedia',
    Peluang_CS: 38.0,
    Diff_Attack_Team: 14.5,
    Diff_Defense_Team: 6.2,
    xMins_Pts: 1.92,
    xG_Pred: 0.62,
    xA_Pred: 0.48,
    xSaves_Pts: 0.0,
    xDC_Pts: 0.25,
    xCS_Pts: 0.38,
    xBP: 1.30,
    Avg_Mins_L5M: 87.0
  },
  {
    id: 5,
    Nama_Pemain: 'Alexander Isak',
    Klub: 'Newcastle',
    Posisi: 'FWD',
    Harga: 8.5,
    Lawan_GW_Berikutnya: 'BOU (A)',
    Status: 'Tersedia',
    Peluang_CS: 29.5,
    Diff_Attack_Team: 8.5,
    Diff_Defense_Team: 4.0,
    xMins_Pts: 1.88,
    xG_Pred: 0.58,
    xA_Pred: 0.18,
    xSaves_Pts: 0.0,
    xDC_Pts: 0.10,
    xCS_Pts: 0.0,
    xBP: 0.95,
    Avg_Mins_L5M: 84.0
  },
  {
    id: 6,
    Nama_Pemain: 'Gabriel Magalhães',
    Klub: 'Arsenal',
    Posisi: 'DEF',
    Harga: 6.2,
    Lawan_GW_Berikutnya: 'SOU (H)',
    Status: 'Tersedia',
    Peluang_CS: 51.5,
    Diff_Attack_Team: 24.5,
    Diff_Defense_Team: 22.0,
    xMins_Pts: 2.00,
    xG_Pred: 0.18,
    xA_Pred: 0.05,
    xSaves_Pts: 0.0,
    xDC_Pts: 1.25,
    xCS_Pts: 2.06,
    xBP: 0.70,
    Avg_Mins_L5M: 90.0
  },
  {
    id: 7,
    Nama_Pemain: 'Trent Alexander-Arnold',
    Klub: 'Liverpool',
    Posisi: 'DEF',
    Harga: 7.1,
    Lawan_GW_Berikutnya: 'EVE (H)',
    Status: 'Tersedia',
    Peluang_CS: 42.0,
    Diff_Attack_Team: 21.0,
    Diff_Defense_Team: 12.8,
    xMins_Pts: 1.85,
    xG_Pred: 0.12,
    xA_Pred: 0.42,
    xSaves_Pts: 0.0,
    xDC_Pts: 0.95,
    xCS_Pts: 1.68,
    xBP: 0.85,
    Avg_Mins_L5M: 82.0
  },
  {
    id: 8,
    Nama_Pemain: 'Ollie Watkins',
    Klub: 'Aston Villa',
    Posisi: 'FWD',
    Harga: 9.1,
    Lawan_GW_Berikutnya: 'MUN (H)',
    Status: 'Tersedia',
    Peluang_CS: 34.0,
    Diff_Attack_Team: 11.2,
    Diff_Defense_Team: 7.5,
    xMins_Pts: 1.90,
    xG_Pred: 0.54,
    xA_Pred: 0.28,
    xSaves_Pts: 0.0,
    xDC_Pts: 0.20,
    xCS_Pts: 0.0,
    xBP: 0.88,
    Avg_Mins_L5M: 85.5
  },
  {
    id: 9,
    Nama_Pemain: 'David Raya',
    Klub: 'Arsenal',
    Posisi: 'GK',
    Harga: 5.6,
    Lawan_GW_Berikutnya: 'SOU (H)',
    Status: 'Tersedia',
    Peluang_CS: 51.5,
    Diff_Attack_Team: 24.5,
    Diff_Defense_Team: 22.0,
    xMins_Pts: 2.00,
    xG_Pred: 0.00,
    xA_Pred: 0.00,
    xSaves_Pts: 1.10,
    xDC_Pts: 0.0,
    xCS_Pts: 2.06,
    xBP: 0.40,
    Avg_Mins_L5M: 90.0
  },
  {
    id: 10,
    Nama_Pemain: 'Josko Gvardiol',
    Klub: 'Man City',
    Posisi: 'DEF',
    Harga: 6.0,
    Lawan_GW_Berikutnya: 'BUR (H)',
    Status: 'Tersedia',
    Peluang_CS: 44.5,
    Diff_Attack_Team: 18.2,
    Diff_Defense_Team: 16.5,
    xMins_Pts: 1.90,
    xG_Pred: 0.14,
    xA_Pred: 0.16,
    xSaves_Pts: 0.0,
    xDC_Pts: 0.80,
    xCS_Pts: 1.78,
    xBP: 0.55,
    Avg_Mins_L5M: 85.0
  }
];

interface ModelBPredictionViewProps {
  players?: PlayerOptionB[];
  onMultipliersChange?: (newMultipliers: PositionMultipliers) => void;
}

export const ModelBPredictionView: React.FC<ModelBPredictionViewProps> = ({
  players = BENCHMARK_PLAYERS,
  onMultipliersChange
}) => {
  const [multipliers, setMultipliers] = useState<PositionMultipliers>(DEFAULT_MULTIPLIERS);
  const [isWeightedActive, setIsWeightedActive] = useState<boolean>(true);
  const [selectedPosFilter, setSelectedPosFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [showConfigPanel, setShowConfigPanel] = useState<boolean>(true);
  const [sortField, setSortField] = useState<'xPoin_OptB' | 'xG_Pts' | 'xA_Pts' | 'Harga'>('xPoin_OptB');
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  // Handle individual multiplier sliders
  const handleMultiplierChange = (pos: 'FWD' | 'MID' | 'DEF' | 'GK', type: 'xG' | 'xA', val: number) => {
    const updated = {
      ...multipliers,
      [pos]: {
        ...multipliers[pos],
        [type]: parseFloat(val.toFixed(2))
      }
    };
    setMultipliers(updated);
    if (onMultipliersChange) {
      onMultipliersChange(updated);
    }
  };

  const handleResetMultipliers = () => {
    setMultipliers(DEFAULT_MULTIPLIERS);
    if (onMultipliersChange) {
      onMultipliersChange(DEFAULT_MULTIPLIERS);
    }
  };

  // Recalculate normalized xG/xA points and total xPoin (Option B) for each player dynamically
  const computedPlayers = useMemo(() => {
    return players.map((p) => {
      const pos = p.Posisi;
      const xgMult = isWeightedActive ? (multipliers[pos]?.xG ?? 1.0) : 1.0;
      const xaMult = isWeightedActive ? (multipliers[pos]?.xA ?? 1.0) : 1.0;

      const goalPoints = GOAL_POINTS_MAP[pos] ?? 4.0;
      const xG_Pts = parseFloat((p.xG_Pred * goalPoints * xgMult).toFixed(2));
      const xA_Pts = parseFloat((p.xA_Pred * 3.0 * xaMult).toFixed(2));

      const xPoin_OptB = parseFloat(
        (
          p.xMins_Pts +
          xG_Pts +
          xA_Pts +
          p.xSaves_Pts +
          p.xDC_Pts +
          p.xCS_Pts +
          p.xBP
        ).toFixed(2)
      );

      // Raw unadjusted points for comparison
      const raw_xG_Pts = parseFloat((p.xG_Pred * goalPoints).toFixed(2));
      const raw_xA_Pts = parseFloat((p.xA_Pred * 3.0).toFixed(2));
      const raw_xPoin_OptB = parseFloat(
        (
          p.xMins_Pts +
          raw_xG_Pts +
          raw_xA_Pts +
          p.xSaves_Pts +
          p.xDC_Pts +
          p.xCS_Pts +
          p.xBP
        ).toFixed(2)
      );

      const delta = parseFloat((xPoin_OptB - raw_xPoin_OptB).toFixed(2));

      return {
        ...p,
        xgMult,
        xaMult,
        xG_Pts,
        xA_Pts,
        xPoin_OptB,
        raw_xPoin_OptB,
        delta
      };
    });
  }, [players, multipliers, isWeightedActive]);

  // Filter & Search
  const filteredAndSortedPlayers = useMemo(() => {
    let result = [...computedPlayers];

    if (selectedPosFilter !== 'ALL') {
      result = result.filter((p) => p.Posisi === selectedPosFilter);
    }

    if (searchQuery.trim() !== '') {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (p) =>
          p.Nama_Pemain.toLowerCase().includes(query) ||
          p.Klub.toLowerCase().includes(query)
      );
    }

    result.sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];
      if (valA < valB) return sortAsc ? -1 : 1;
      if (valA > valB) return sortAsc ? 1 : -1;
      return 0;
    });

    return result;
  }, [computedPlayers, selectedPosFilter, searchQuery, sortField, sortAsc]);

  const toggleSort = (field: 'xPoin_OptB' | 'xG_Pts' | 'xA_Pts' | 'Harga') => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-4 md:p-6 bg-slate-50 min-h-screen text-slate-800 font-sans">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 md:p-6 shadow-sm mb-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="bg-indigo-100 text-indigo-800 text-xs font-semibold px-2.5 py-0.8 rounded-full border border-indigo-200">
                Option B Engine
              </span>
              <span className="bg-emerald-100 text-emerald-800 text-xs font-semibold px-2.5 py-0.8 rounded-full border border-emerald-200">
                Normalized Points Formula
              </span>
            </div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-900">
              Model B: Bottom-Up Prediction & Multiplier Calibration
            </h1>
            <p className="text-slate-600 text-sm mt-1 max-w-3xl">
              Dekonstruksi komponen xPoin matchday individual dengan normalisasi efisiensi konversi posisi (<code className="text-indigo-600 font-mono">Goal &gt; xG</code> dan <code className="text-indigo-600 font-mono">Assist &gt; xA</code>). Sesuaikan weighted multiplier per posisi secara interaktif di bawah.
            </p>
          </div>

          {/* Master Toggle */}
          <div className="flex items-center gap-3 bg-slate-100 p-2 rounded-lg border border-slate-200 self-start md:self-auto">
            <span className="text-xs font-medium text-slate-700">Weighted Multiplier:</span>
            <button
              onClick={() => setIsWeightedActive(!isWeightedActive)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 ${
                isWeightedActive ? 'bg-indigo-600' : 'bg-slate-300'
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  isWeightedActive ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
            <span className={`text-xs font-bold ${isWeightedActive ? 'text-indigo-600' : 'text-slate-400'}`}>
              {isWeightedActive ? 'AKTIF' : 'NONAKTIF (1.0x)'}
            </span>
          </div>
        </div>
      </div>

      {/* Interactive Weighted Multiplier Calibration Panel */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm mb-6 overflow-hidden">
        <div
          className="flex items-center justify-between p-4 bg-slate-50/80 border-b border-slate-200 cursor-pointer select-none"
          onClick={() => setShowConfigPanel(!showConfigPanel)}
        >
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-5 h-5 text-indigo-600" />
            <h2 className="font-semibold text-slate-900 text-base">
              Kalibrasi Weighted Multiplier Posisi (Historical Efficiency Tuning)
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleResetMultipliers();
              }}
              className="flex items-center gap-1 text-xs text-slate-600 hover:text-indigo-600 bg-white border border-slate-200 px-2.5 py-1 rounded shadow-xs hover:border-indigo-300 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Default
            </button>
            <ChevronDown
              className={`w-5 h-5 text-slate-400 transition-transform ${
                showConfigPanel ? 'transform rotate-180' : ''
              }`}
            />
          </div>
        </div>

        {showConfigPanel && (
          <div className="p-5">
            <div className="mb-4 text-xs text-slate-500 flex items-start gap-1.5 bg-indigo-50/60 p-3 rounded-lg border border-indigo-100">
              <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
              <div>
                <strong>Formula Kontribusi Poin Ternormalisasi:</strong>
                <br />
                <span className="font-mono text-indigo-900">
                  xG Pts = xG Pred * Poin Gol (GK:10, DEF:6, MID:5, FWD:4) * Pos_xG_Mult
                </span>
                <br />
                <span className="font-mono text-indigo-900">
                  xA Pts = xA Pred * 3.0 * Pos_xA_Mult
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* FWD Tuning Card */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-500"></span>
                    <span className="font-bold text-slate-900 text-sm">FWD (Penyerang)</span>
                  </div>
                  <span className="text-xs bg-red-100 text-red-800 font-semibold px-2 py-0.5 rounded">
                    4 pts / Gol
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-600 font-medium">xG Multiplier:</span>
                      <span className="font-mono font-bold text-indigo-700">
                        {multipliers.FWD.xG.toFixed(2)}x
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.70"
                      max="1.40"
                      step="0.01"
                      disabled={!isWeightedActive}
                      value={multipliers.FWD.xG}
                      onChange={(e) => handleMultiplierChange('FWD', 'xG', parseFloat(e.target.value))}
                      className="w-full accent-indigo-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer disabled:opacity-40"
                    />
                    <div className="text-[10px] text-slate-400 mt-0.5">Overperform (+8% default)</div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-600 font-medium">xA Multiplier:</span>
                      <span className="font-mono font-bold text-indigo-700">
                        {multipliers.FWD.xA.toFixed(2)}x
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.70"
                      max="1.40"
                      step="0.01"
                      disabled={!isWeightedActive}
                      value={multipliers.FWD.xA}
                      onChange={(e) => handleMultiplierChange('FWD', 'xA', parseFloat(e.target.value))}
                      className="w-full accent-indigo-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer disabled:opacity-40"
                    />
                    <div className="text-[10px] text-slate-400 mt-0.5">Umpan matang (+5% default)</div>
                  </div>
                </div>
              </div>

              {/* MID Tuning Card */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                    <span className="font-bold text-slate-900 text-sm">MID (Gelandang)</span>
                  </div>
                  <span className="text-xs bg-amber-100 text-amber-800 font-semibold px-2 py-0.5 rounded">
                    5 pts / Gol
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-600 font-medium">xG Multiplier:</span>
                      <span className="font-mono font-bold text-indigo-700">
                        {multipliers.MID.xG.toFixed(2)}x
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.70"
                      max="1.40"
                      step="0.01"
                      disabled={!isWeightedActive}
                      value={multipliers.MID.xG}
                      onChange={(e) => handleMultiplierChange('MID', 'xG', parseFloat(e.target.value))}
                      className="w-full accent-indigo-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer disabled:opacity-40"
                    />
                    <div className="text-[10px] text-slate-400 mt-0.5">Inside forward (+4% default)</div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-600 font-medium">xA Multiplier:</span>
                      <span className="font-mono font-bold text-indigo-700">
                        {multipliers.MID.xA.toFixed(2)}x
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.70"
                      max="1.40"
                      step="0.01"
                      disabled={!isWeightedActive}
                      value={multipliers.MID.xA}
                      onChange={(e) => handleMultiplierChange('MID', 'xA', parseFloat(e.target.value))}
                      className="w-full accent-indigo-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer disabled:opacity-40"
                    />
                    <div className="text-[10px] text-slate-400 mt-0.5">Playmakers (+6% default)</div>
                  </div>
                </div>
              </div>

              {/* DEF Tuning Card */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                    <span className="font-bold text-slate-900 text-sm">DEF (Bek)</span>
                  </div>
                  <span className="text-xs bg-blue-100 text-blue-800 font-semibold px-2 py-0.5 rounded">
                    6 pts / Gol
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-600 font-medium">xG Multiplier:</span>
                      <span className="font-mono font-bold text-indigo-700">
                        {multipliers.DEF.xG.toFixed(2)}x
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.50"
                      max="1.30"
                      step="0.01"
                      disabled={!isWeightedActive}
                      value={multipliers.DEF.xG}
                      onChange={(e) => handleMultiplierChange('DEF', 'xG', parseFloat(e.target.value))}
                      className="w-full accent-indigo-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer disabled:opacity-40"
                    />
                    <div className="text-[10px] text-slate-400 mt-0.5">Low-conv set piece (-12% default)</div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-600 font-medium">xA Multiplier:</span>
                      <span className="font-mono font-bold text-indigo-700">
                        {multipliers.DEF.xA.toFixed(2)}x
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.50"
                      max="1.30"
                      step="0.01"
                      disabled={!isWeightedActive}
                      value={multipliers.DEF.xA}
                      onChange={(e) => handleMultiplierChange('DEF', 'xA', parseFloat(e.target.value))}
                      className="w-full accent-indigo-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer disabled:opacity-40"
                    />
                    <div className="text-[10px] text-slate-400 mt-0.5">Deep crosses (-8% default)</div>
                  </div>
                </div>
              </div>

              {/* GK Tuning Card */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                    <span className="font-bold text-slate-900 text-sm">GK (Kiper)</span>
                  </div>
                  <span className="text-xs bg-emerald-100 text-emerald-800 font-semibold px-2 py-0.5 rounded">
                    10 pts / Gol
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-600 font-medium">xG Multiplier:</span>
                      <span className="font-mono font-bold text-slate-400">0.00x</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="0"
                      disabled
                      value="0"
                      className="w-full h-1.5 bg-slate-200 rounded-lg opacity-40 cursor-not-allowed"
                    />
                    <div className="text-[10px] text-slate-400 mt-0.5">Diproteksi 0.00x untuk kiper</div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-600 font-medium">xA Multiplier:</span>
                      <span className="font-mono font-bold text-indigo-700">
                        {multipliers.GK.xA.toFixed(2)}x
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.00"
                      max="1.00"
                      step="0.05"
                      disabled={!isWeightedActive}
                      value={multipliers.GK.xA}
                      onChange={(e) => handleMultiplierChange('GK', 'xA', parseFloat(e.target.value))}
                      className="w-full accent-indigo-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer disabled:opacity-40"
                    />
                    <div className="text-[10px] text-slate-400 mt-0.5">Long-pass assists (0.50x default)</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Visualisasi Group by Posisi: xG vs G & xA vs A */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs mb-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 mb-4 border-b border-slate-100 gap-2">
          <div>
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-indigo-600" />
              <h2 className="font-bold text-slate-900 text-lg">
                Analisis Efisiensi: Komparasi xG vs Gol (G) & xA vs Asis (A) Group By Posisi
              </h2>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Evaluasi agregat tren konversi penyelesaian akhir (<code className="text-indigo-600">Goal vs xG</code>) dan kreasi peluang (<code className="text-amber-600">Assist vs xA</code>) untuk seluruh posisi pemain.
            </p>
          </div>
          <span className="self-start md:self-auto text-[11px] font-semibold bg-slate-100 text-slate-600 px-2.5 py-1 rounded-full border border-slate-200">
            Dataset FPL 2026 Agregat
          </span>
        </div>

        {/* Position Summary Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {(['FWD', 'MID', 'DEF', 'GK'] as const).map((pos) => {
            const posStats = {
              FWD: { g: 36, xg: 47.47, a: 20, xa: 6.49, color: 'border-l-red-500', badge: 'bg-red-100 text-red-800' },
              MID: { g: 77, xg: 77.52, a: 78, xa: 60.06, color: 'border-l-amber-500', badge: 'bg-amber-100 text-amber-800' },
              DEF: { g: 21, xg: 28.08, a: 31, xa: 25.39, color: 'border-l-blue-500', badge: 'bg-blue-100 text-blue-800' },
              GK: { g: 0, xg: 0.00, a: 1, xa: 0.33, color: 'border-l-emerald-500', badge: 'bg-emerald-100 text-emerald-800' },
            }[pos];

            const deltaG = posStats.g - posStats.xg;
            const deltaA = posStats.a - posStats.xa;
            const ratioG = posStats.xg > 0 ? (posStats.g / posStats.xg).toFixed(2) : '0.00';
            const ratioA = posStats.xa > 0 ? (posStats.a / posStats.xa).toFixed(2) : '0.00';

            return (
              <div key={pos} className={`bg-slate-50 border border-slate-200 border-l-4 ${posStats.color} rounded-lg p-3.5 shadow-2xs`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-xs font-bold px-2 py-0.5 rounded ${posStats.badge}`}>
                    Posisi: {pos}
                  </span>
                  <span className="text-[11px] font-mono text-slate-500">
                    G/xG: <b>{ratioG}x</b>
                  </span>
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Total Gol (G):</span>
                    <span className="font-semibold text-slate-900 font-mono">
                      {posStats.g} <span className="text-slate-400 font-normal">/ xG {posStats.xg.toFixed(1)}</span>
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-[11px]">
                    <span className="text-slate-500">Selisih (G - xG):</span>
                    <span className={`font-mono font-bold ${deltaG >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                      {deltaG >= 0 ? `+${deltaG.toFixed(1)}` : deltaG.toFixed(1)}
                    </span>
                  </div>

                  <div className="pt-1 border-t border-slate-200/60 flex justify-between items-center">
                    <span className="text-slate-600">Total Asis (A):</span>
                    <span className="font-semibold text-slate-900 font-mono">
                      {posStats.a} <span className="text-slate-400 font-normal">/ xA {posStats.xa.toFixed(1)}</span>
                    </span>
                  </div>
                  <div className="flex justify-between items-center text-[11px]">
                    <span className="text-slate-500">Selisih (A - xA):</span>
                    <span className={`font-mono font-bold ${deltaA >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                      {deltaA >= 0 ? `+${deltaA.toFixed(1)}` : deltaA.toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Dual Bar Comparison Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chart 1: xG vs Gol */}
          <div className="bg-slate-50/70 border border-slate-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-1.5">
                <Target className="w-4 h-4 text-indigo-600" />
                <h3 className="font-bold text-slate-800 text-sm">⚽ Komparasi Expected Goals (xG) vs Aktual Gol (G)</h3>
              </div>
              <div className="flex items-center gap-3 text-[11px]">
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-indigo-300"></span> xG
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-indigo-600"></span> Gol (G)
                </span>
              </div>
            </div>

            <div className="space-y-3 pt-2">
              {[
                { pos: 'FWD', g: 36, xg: 47.47, max: 90 },
                { pos: 'MID', g: 77, xg: 77.52, max: 90 },
                { pos: 'DEF', g: 21, xg: 28.08, max: 90 },
                { pos: 'GK', g: 0, xg: 0.00, max: 90 }
              ].map((item) => (
                <div key={item.pos} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="font-bold text-slate-700 w-10">{item.pos}</span>
                    <div className="flex items-center gap-3 font-mono text-[11px]">
                      <span className="text-indigo-900 font-semibold">{item.g} Gol</span>
                      <span className="text-slate-400">vs</span>
                      <span className="text-indigo-400">{item.xg.toFixed(1)} xG</span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {/* xG bar */}
                    <div className="w-full bg-slate-200 h-3 rounded overflow-hidden">
                      <div
                        className="bg-indigo-300 h-full rounded transition-all duration-500"
                        style={{ width: `${(item.xg / item.max) * 100}%` }}
                      />
                    </div>
                    {/* Goal bar */}
                    <div className="w-full bg-slate-200 h-3 rounded overflow-hidden">
                      <div
                        className="bg-indigo-600 h-full rounded transition-all duration-500"
                        style={{ width: `${(item.g / item.max) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-2 border-t border-slate-200 text-[11px] text-slate-500 flex justify-between">
              <span>*MID dan FWD menyumbang &gt;85% output gol liga</span>
              <span className="font-medium text-indigo-700">Rasio tertinggi: MID (0.99x)</span>
            </div>
          </div>

          {/* Chart 2: xA vs Asis */}
          <div className="bg-slate-50/70 border border-slate-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-amber-600" />
                <h3 className="font-bold text-slate-800 text-sm">🎯 Komparasi Expected Assists (xA) vs Aktual Asis (A)</h3>
              </div>
              <div className="flex items-center gap-3 text-[11px]">
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-amber-200"></span> xA
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-amber-600"></span> Asis (A)
                </span>
              </div>
            </div>

            <div className="space-y-3 pt-2">
              {[
                { pos: 'FWD', a: 20, xa: 6.49, max: 90 },
                { pos: 'MID', a: 78, xa: 60.06, max: 90 },
                { pos: 'DEF', a: 31, xa: 25.39, max: 90 },
                { pos: 'GK', a: 1, xa: 0.33, max: 90 }
              ].map((item) => (
                <div key={item.pos} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="font-bold text-slate-700 w-10">{item.pos}</span>
                    <div className="flex items-center gap-3 font-mono text-[11px]">
                      <span className="text-amber-900 font-semibold">{item.a} Asis</span>
                      <span className="text-slate-400">vs</span>
                      <span className="text-amber-400">{item.xa.toFixed(1)} xA</span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {/* xA bar */}
                    <div className="w-full bg-slate-200 h-3 rounded overflow-hidden">
                      <div
                        className="bg-amber-200 h-full rounded transition-all duration-500"
                        style={{ width: `${(item.xa / item.max) * 100}%` }}
                      />
                    </div>
                    {/* Assist bar */}
                    <div className="w-full bg-slate-200 h-3 rounded overflow-hidden">
                      <div
                        className="bg-amber-600 h-full rounded transition-all duration-500"
                        style={{ width: `${(item.a / item.max) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-2 border-t border-slate-200 text-[11px] text-slate-500 flex justify-between">
              <span>*Playmaker MID menghasilkan konversi peluang tertinggi</span>
              <span className="font-medium text-amber-700">Rasio Asis: MID (+17.9 over xA)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 mb-4">
        {/* Position Pills */}
        <div className="flex items-center gap-1.5 bg-slate-200/70 p-1 rounded-lg w-full sm:w-auto">
          {['ALL', 'FWD', 'MID', 'DEF', 'GK'].map((pos) => (
            <button
              key={pos}
              onClick={() => setSelectedPosFilter(pos)}
              className={`px-3 py-1 text-xs font-semibold rounded-md transition-all ${
                selectedPosFilter === pos
                  ? 'bg-white text-indigo-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {pos === 'ALL' ? 'Semua Posisi' : pos}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="relative w-full sm:w-64">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
          <input
            type="text"
            placeholder="Cari pemain atau klub..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-slate-800 placeholder-slate-400"
          />
        </div>
      </div>

      {/* Prediction Results Table */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-xs overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Pemain & Klub</th>
                <th className="py-3 px-3">Posisi</th>
                <th className="py-3 px-3 cursor-pointer hover:bg-slate-100" onClick={() => toggleSort('Harga')}>
                  <div className="flex items-center gap-1">
                    <span>Harga</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th className="py-3 px-3">Lawan GW</th>
                <th
                  className="py-3 px-4 bg-indigo-50/60 text-indigo-950 font-bold cursor-pointer hover:bg-indigo-100/60"
                  onClick={() => toggleSort('xPoin_OptB')}
                >
                  <div className="flex items-center gap-1">
                    <span>xPoin (Option B)</span>
                    <ArrowUpDown className="w-3.5 h-3.5 text-indigo-600" />
                  </div>
                </th>
                <th className="py-3 px-3 text-center">xG Mult</th>
                <th className="py-3 px-3 text-center">xA Mult</th>
                <th
                  className="py-3 px-3 text-right cursor-pointer hover:bg-slate-100"
                  onClick={() => toggleSort('xG_Pts')}
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>xG Pts</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th
                  className="py-3 px-3 text-right cursor-pointer hover:bg-slate-100"
                  onClick={() => toggleSort('xA_Pts')}
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>xA Pts</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th className="py-3 px-3 text-right">xMins Pts</th>
                <th className="py-3 px-3 text-right">xCS Pts</th>
                <th className="py-3 px-3 text-right">xDC Pts</th>
                <th className="py-3 px-3 text-right">xBP</th>
                <th className="py-3 px-3 text-center">Dixon-Coles CS%</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {filteredAndSortedPlayers.map((player) => {
                const isPositiveDelta = player.delta > 0;
                const isZeroDelta = player.delta === 0;

                return (
                  <tr key={player.id} className="hover:bg-slate-50/80 transition-colors">
                    {/* Name & Club */}
                    <td className="py-3 px-4">
                      <div className="font-semibold text-slate-900">{player.Nama_Pemain}</div>
                      <div className="text-[11px] text-slate-500">{player.Klub}</div>
                    </td>

                    {/* Position */}
                    <td className="py-3 px-3">
                      <span
                        className={`inline-block px-2 py-0.5 text-[10px] font-bold rounded ${
                          player.Posisi === 'FWD'
                            ? 'bg-red-100 text-red-800'
                            : player.Posisi === 'MID'
                            ? 'bg-amber-100 text-amber-800'
                            : player.Posisi === 'DEF'
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-emerald-100 text-emerald-800'
                        }`}
                      >
                        {player.Posisi}
                      </span>
                    </td>

                    {/* Price */}
                    <td className="py-3 px-3 font-mono">£{player.Harga.toFixed(1)}m</td>

                    {/* Opponent */}
                    <td className="py-3 px-3 text-slate-600 font-medium">
                      {player.Lawan_GW_Berikutnya}
                    </td>

                    {/* Main Option B Points */}
                    <td className="py-3 px-4 bg-indigo-50/40">
                      <div className="flex items-center gap-1.5">
                        <span className="text-base font-extrabold text-indigo-700 font-mono">
                          {player.xPoin_OptB.toFixed(2)}
                        </span>
                        {isWeightedActive && !isZeroDelta && (
                          <span
                            className={`text-[10px] font-semibold px-1 py-0.2 rounded font-mono ${
                              isPositiveDelta
                                ? 'bg-emerald-100 text-emerald-800'
                                : 'bg-rose-100 text-rose-800'
                            }`}
                          >
                            {isPositiveDelta ? `+${player.delta.toFixed(2)}` : `${player.delta.toFixed(2)}`}
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-slate-400">
                        Raw: {player.raw_xPoin_OptB.toFixed(2)} pts
                      </div>
                    </td>

                    {/* Multipliers */}
                    <td className="py-3 px-3 text-center">
                      <span
                        className={`font-mono text-[11px] px-1.5 py-0.5 rounded ${
                          player.xgMult > 1.0
                            ? 'bg-emerald-50 text-emerald-700 font-bold border border-emerald-200'
                            : player.xgMult < 1.0
                            ? 'bg-amber-50 text-amber-700 font-bold border border-amber-200'
                            : 'bg-slate-100 text-slate-600'
                        }`}
                      >
                        {player.xgMult.toFixed(2)}x
                      </span>
                    </td>
                    <td className="py-3 px-3 text-center">
                      <span
                        className={`font-mono text-[11px] px-1.5 py-0.5 rounded ${
                          player.xaMult > 1.0
                            ? 'bg-emerald-50 text-emerald-700 font-bold border border-emerald-200'
                            : player.xaMult < 1.0
                            ? 'bg-amber-50 text-amber-700 font-bold border border-amber-200'
                            : 'bg-slate-100 text-slate-600'
                        }`}
                      >
                        {player.xaMult.toFixed(2)}x
                      </span>
                    </td>

                    {/* Components */}
                    <td className="py-3 px-3 text-right font-mono font-medium text-slate-900">
                      {player.xG_Pts.toFixed(2)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono font-medium text-slate-900">
                      {player.xA_Pts.toFixed(2)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-slate-600">
                      {player.xMins_Pts.toFixed(2)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-slate-600">
                      {player.xCS_Pts.toFixed(2)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-slate-600">
                      {player.xDC_Pts.toFixed(2)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-slate-600">
                      {player.xBP.toFixed(2)}
                    </td>

                    {/* Clean Sheet % */}
                    <td className="py-3 px-3 text-center">
                      <div className="inline-flex items-center gap-1.5">
                        <div className="w-12 bg-slate-200 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-indigo-600 h-1.5 rounded-full"
                            style={{ width: `${Math.min(100, player.Peluang_CS)}%` }}
                          />
                        </div>
                        <span className="text-[11px] font-mono font-semibold text-slate-700">
                          {player.Peluang_CS.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Footer info */}
        <div className="p-3 bg-slate-50 border-t border-slate-200 text-slate-500 text-xs flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            Menampilkan <span className="font-semibold text-slate-800">{filteredAndSortedPlayers.length}</span> pemain.
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Multiplier &gt; 1.0 (Boost)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-amber-500"></span> Multiplier &lt; 1.0 (Discount)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-indigo-600"></span> xPoin Option B
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelBPredictionView;
