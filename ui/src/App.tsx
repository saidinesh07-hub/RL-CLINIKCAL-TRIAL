import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Zap, TrendingUp, Users, Brain, Activity } from 'lucide-react';

interface EpisodeData {
  episode: number;
  reward: number;
  assignmentRate: number;
  diversityScore: number;
}

interface SimulationResponse {
  score: number;
  assignmentRate: number;
  diversity: number;
  fillRate: number;
  reward: number;
}

// Animated counter component
const AnimatedCounter = ({ value, decimals = 3 }: { value: number; decimals?: number }) => {
  const [displayValue, setDisplayValue] = useState(0);
  useEffect(() => {
    const duration = 1000;
    const start = Date.now();
    const diff = value - displayValue;
    
    const interval = setInterval(() => {
      const now = Date.now();
      const progress = Math.min((now - start) / duration, 1);
      setDisplayValue(displayValue + diff * progress);
      if (progress === 1) clearInterval(interval);
    }, 30);
    
    return () => clearInterval(interval);
  }, [value, displayValue]);
  
  return <>{displayValue.toFixed(decimals)}</>;
};

// Cursor-tracking AI Avatar
const AIAvatar = () => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const avatarRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);
  
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden">
      <motion.div
        ref={avatarRef}
        animate={{
          x: mousePosition.x - 20,
          y: mousePosition.y - 20,
        }}
        transition={{ type: 'spring', damping: 20, mass: 0.5 }}
        className="fixed w-10 h-10 pointer-events-none"
      >
        <motion.div
          animate={{ y: [0, -8, 0] }}
          transition={{ duration: 3, repeat: Infinity }}
          className="relative w-full h-full"
        >
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-400 to-purple-500 blur-lg opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-400 to-purple-500 flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

function App() {
  const [simulationData, setSimulationData] = useState<EpisodeData[]>([]);
  const [finalMetrics, setFinalMetrics] = useState<SimulationResponse['finalMetrics'] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [hasSimulated, setHasSimulated] = useState(false);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/run-simulation');
      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }
      const data = (await response.json()) as EpisodeData[];
      console.log('Backend response:', data);
      setSimulationData(data);
      setHasSimulated(true);
      if (data.length > 0) {
        const avgReward = data.reduce((sum, ep) => sum + ep.reward, 0) / data.length;
        const avgAssignmentRate = data.reduce((sum, ep) => sum + ep.assignmentRate, 0) / data.length;
        const avgDiversity = data.reduce((sum, ep) => sum + ep.diversityScore, 0) / data.length;
        setFinalMetrics({
          score: 0.85,
          assignmentRate: avgAssignmentRate,
          diversity: avgDiversity,
          fillRate: 0.68,
          reward: avgReward
        });
      } else {
        setFinalMetrics(null);
      }
      setConnected(true);
      console.log('simulationData:', data);
    } catch (err: any) {
      setError(err?.message || 'Failed to fetch data');
      setSimulationData([]);
      setFinalMetrics(null);
      setConnected(false);
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const previewData = simulationData.slice(0, 10);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.15, delayChildren: 0.2 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-black text-slate-100">
      {/* Enhanced animated background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-blue-950/20 to-purple-950/20" />
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
          className="absolute -top-96 -right-96 w-96 h-96 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 rounded-full blur-3xl"
        />
        <motion.div
          animate={{ rotate: -360 }}
          transition={{ duration: 25, repeat: Infinity, ease: 'linear' }}
          className="absolute -bottom-96 -left-96 w-96 h-96 bg-gradient-to-tr from-purple-500/10 to-pink-500/10 rounded-full blur-3xl"
        />
      </div>

      <AIAvatar />

      <motion.div
        className="relative mx-auto max-w-7xl px-6 py-12 sm:px-8 lg:px-10"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* HERO SECTION */}
        <motion.section variants={itemVariants} className="mb-16">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="rounded-[40px] border border-white/10 bg-gradient-to-br from-slate-900/40 via-blue-900/20 to-purple-900/20 p-12 backdrop-blur-xl shadow-[0_25px_80px_-35px_rgba(15,23,42,0.9)]"
          >
            <div className="flex flex-col items-center justify-center text-center gap-6">
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.1 }}
                className="flex flex-col items-center justify-center text-center gap-4 max-w-3xl"
              >
                <p className="text-xl md:text-2xl text-slate-400 tracking-wide font-light">
                  Clinical Trial
                </p>
                <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold leading-tight drop-shadow-[0_0_25px_rgba(139,92,246,0.5)]">
                  <span className="bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
                    Optimization AI
                  </span>
                </h1>
                <p className="text-sm md:text-lg text-slate-400 max-w-2xl leading-relaxed mt-4">
                  AI-powered reinforcement learning for intelligent patient assignment and fairness
                </p>
              </motion.div>

              <motion.button
                onClick={runSimulation}
                disabled={loading}
                whileHover={{ scale: loading ? 1 : 1.05 }}
                whileTap={{ scale: loading ? 1 : 0.95 }}
                className="group relative rounded-full bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 px-10 py-4 text-lg font-bold text-white shadow-[0_20px_70px_-20px_rgba(6,182,212,0.6)] transition-all duration-300 disabled:opacity-70 hover:shadow-[0_25px_90px_-20px_rgba(6,182,212,0.8)]"
              >
                <span className="relative inline-flex items-center gap-3">
                  {loading ? (
                    <>
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="w-5 h-5 flex items-center justify-center"
                      >
                        <Zap className="w-5 h-5" />
                      </motion.div>
                      <span>Running RL Simulation...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-5 h-5" />
                      <span>Run Simulation</span>
                    </>
                  )}
                </span>
                {!loading && (
                  <motion.div
                    className="absolute inset-0 rounded-full bg-white/20 blur-xl opacity-0 group-hover:opacity-100"
                    transition={{ duration: 0.3 }}
                  />
                )}
              </motion.button>
            </div>
          </motion.div>
        </motion.section>

        {/* METRICS CARDS */}
        <motion.section
          variants={itemVariants}
          className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 mb-12"
        >
          {[
            {
              icon: TrendingUp,
              label: 'Mean Reward',
              value: finalMetrics?.reward ?? 0,
              color: 'from-cyan-500 to-blue-500',
              textColor: 'text-cyan-300',
              borderColor: 'border-cyan-500/20 hover:border-cyan-500/50'
            },
            {
              icon: Users,
              label: 'Assignment Rate',
              value: finalMetrics?.assignmentRate ?? 0,
              color: 'from-emerald-500 to-cyan-500',
              textColor: 'text-emerald-300',
              borderColor: 'border-emerald-500/20 hover:border-emerald-500/50'
            },
            {
              icon: Activity,
              label: 'Diversity Score',
              value: finalMetrics?.diversity ?? 0,
              color: 'from-purple-500 to-pink-500',
              textColor: 'text-purple-300',
              borderColor: 'border-purple-500/20 hover:border-purple-500/50'
            },
            {
              icon: Brain,
              label: 'Backend Status',
              value: connected ? 1 : 0,
              showAsText: true,
              textValue: connected ? 'Connected' : loading ? 'Connecting' : 'Offline',
              color: 'from-pink-500 to-purple-500',
              textColor: connected ? 'text-emerald-300' : 'text-rose-300',
              borderColor: 'border-pink-500/20 hover:border-pink-500/50'
            }
          ].map((metric, idx) => {
            const Icon = metric.icon;
            return (
              <motion.div
                key={idx}
                variants={itemVariants}
                whileHover={{ y: -8, boxShadow: '0 20px 40px -20px rgba(0,0,0,0.5)' }}
                className={`group rounded-2xl border ${metric.borderColor} bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur-xl transition-all duration-300`}
              >
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${metric.color} p-2.5 mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon className="w-full h-full text-white" />
                </div>
                <p className="text-sm uppercase tracking-[0.3em] text-slate-400 font-semibold">{metric.label}</p>
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className={`mt-4 text-4xl font-bold ${metric.textColor}`}
                >
                  {metric.showAsText ? (
                    metric.textValue
                  ) : (
                    hasSimulated ? <AnimatedCounter value={metric.value} /> : '--'
                  )}
                </motion.p>
              </motion.div>
            );
          })}
        </motion.section>

        {/* ERROR SECTION */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-8 rounded-2xl border border-rose-500/30 bg-rose-500/5 p-6 backdrop-blur-xl"
          >
            <p className="font-semibold text-rose-300">⚠ Error:</p>
            <p className="mt-2 text-sm text-rose-200">{error}</p>
          </motion.div>
        )}

        {/* CHARTS SECTION */}
        {simulationData.length > 0 ? (
          <motion.section
            variants={itemVariants}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12"
          >
            {/* Reward Chart */}
            <motion.div
              variants={itemVariants}
              whileHover={{ borderColor: 'rgba(6,182,212,0.5)' }}
              className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/40 via-slate-900/20 to-slate-800/20 p-8 backdrop-blur-xl transition-all duration-300"
            >
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-white">Reward Progress</h2>
                  <p className="mt-2 text-sm text-slate-400">Episode-wise training rewards</p>
                </div>
                <motion.span
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-300"
                >
                  Live
                </motion.span>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={simulationData} margin={{ top: 12, right: 18, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="rewardGradient" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#06b6d4" stopOpacity={1} />
                        <stop offset="100%" stopColor="#ec4899" stopOpacity={1} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="episode" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #475569',
                        borderRadius: 12,
                        boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
                      }}
                      formatter={(value: number) => value.toFixed(2)}
                    />
                    <Line
                      type="monotone"
                      dataKey="reward"
                      stroke="url(#rewardGradient)"
                      strokeWidth={3}
                      dot={false}
                      animationDuration={1000}
                      isAnimationActive={true}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Assignment Rate Chart */}
            <motion.div
              variants={itemVariants}
              whileHover={{ borderColor: 'rgba(34,211,238,0.5)' }}
              className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/40 via-slate-900/20 to-slate-800/20 p-8 backdrop-blur-xl transition-all duration-300"
            >
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-white">Assignment Rate</h2>
                  <p className="mt-2 text-sm text-slate-400">Patient allocation efficiency</p>
                </div>
                <motion.span
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 2, repeat: Infinity, delay: 0.1 }}
                  className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-300"
                >
                  Active
                </motion.span>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={simulationData} margin={{ top: 12, right: 18, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="assignmentGradient" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#10b981" stopOpacity={1} />
                        <stop offset="100%" stopColor="#06b6d4" stopOpacity={1} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="episode" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #475569',
                        borderRadius: 12,
                        boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
                      }}
                      formatter={(value: number) => value.toFixed(3)}
                    />
                    <Line
                      type="monotone"
                      dataKey="assignmentRate"
                      stroke="url(#assignmentGradient)"
                      strokeWidth={3}
                      dot={false}
                      animationDuration={1000}
                      isAnimationActive={true}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Diversity Score Chart */}
            <motion.div
              variants={itemVariants}
              whileHover={{ borderColor: 'rgba(168,85,247,0.5)' }}
              className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/40 via-slate-900/20 to-slate-800/20 p-8 backdrop-blur-xl transition-all duration-300 lg:col-span-2"
            >
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-white">Diversity Score</h2>
                  <p className="mt-2 text-sm text-slate-400">Demographics representation & fairness</p>
                </div>
                <motion.span
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 2, repeat: Infinity, delay: 0.2 }}
                  className="rounded-full border border-purple-500/20 bg-purple-500/10 px-3 py-1 text-xs font-semibold text-purple-300"
                >
                  Monitoring
                </motion.span>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={simulationData} margin={{ top: 12, right: 18, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="diversityGradient" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#a855f7" stopOpacity={1} />
                        <stop offset="100%" stopColor="#ec4899" stopOpacity={1} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="episode" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        border: '1px solid #475569',
                        borderRadius: 12,
                        boxShadow: '0 10px 30px rgba(0,0,0,0.5)'
                      }}
                      formatter={(value: number) => value.toFixed(3)}
                    />
                    <Line
                      type="monotone"
                      dataKey="diversityScore"
                      stroke="url(#diversityGradient)"
                      strokeWidth={3}
                      dot={false}
                      animationDuration={1000}
                      isAnimationActive={true}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          </motion.section>
        ) : !loading && hasSimulated ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/40 to-slate-800/20 p-12 backdrop-blur-xl text-center"
          >
            <p className="text-slate-400">No data available. Run the simulation to view graphs.</p>
          </motion.div>
        ) : null}

        {/* DATA TABLE */}
        {simulationData.length > 0 && (
          <motion.section variants={itemVariants} className="mb-8">
            <motion.div
              whileHover={{ borderColor: 'rgba(255,255,255,0.2)' }}
              className="rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900/40 via-slate-900/20 to-slate-800/20 p-8 backdrop-blur-xl transition-all duration-300"
            >
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-white">Episode Data</h2>
                  <p className="mt-2 text-sm text-slate-400">Preview of the first 10 training episodes</p>
                </div>
                <motion.span
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 4, repeat: Infinity }}
                  className="rounded-full border border-white/20 bg-white/5 px-4 py-2 text-xs uppercase tracking-[0.2em] text-slate-400 font-semibold"
                >
                  {simulationData.length} Episodes
                </motion.span>
              </div>

              <div className="overflow-hidden rounded-2xl border border-white/5">
                <table className="w-full text-sm">
                  <thead className="bg-gradient-to-r from-slate-800/50 to-slate-900/30 border-b border-white/5">
                    <tr>
                      <th className="px-6 py-4 text-left font-semibold text-slate-300">Episode</th>
                      <th className="px-6 py-4 text-left font-semibold text-cyan-300">Reward</th>
                      <th className="px-6 py-4 text-left font-semibold text-emerald-300">Assignment</th>
                      <th className="px-6 py-4 text-left font-semibold text-purple-300">Diversity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {previewData.map((row, idx) => (
                      <motion.tr
                        key={row.episode}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.05 }}
                        whileHover={{ backgroundColor: 'rgba(255,255,255,0.05)' }}
                        className="border-b border-white/5 transition-colors duration-200"
                      >
                        <td className="px-6 py-4 text-slate-400">{row.episode}</td>
                        <td className="px-6 py-4 font-semibold text-cyan-300">{row.reward.toFixed(2)}</td>
                        <td className="px-6 py-4 text-emerald-300">{row.assignmentRate.toFixed(3)}</td>
                        <td className="px-6 py-4 text-purple-300">{row.diversityScore.toFixed(3)}</td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          </motion.section>
        )}
      </motion.div>
    </div>
  );
}

export default App;
