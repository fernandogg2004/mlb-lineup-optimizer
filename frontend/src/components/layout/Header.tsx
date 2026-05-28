export default function Header() {
  return (
    <header
      className="flex items-center justify-between px-6 py-3 border-b"
      style={{ background: '#111827', borderColor: '#1F2937' }}
    >
      {/* Left: Logo */}
      <div className="flex flex-col">
        <span className="text-base font-bold text-white tracking-wide">
          ⚾ MLB War Room
        </span>
        <span className="text-xs text-slate-400 tracking-widest uppercase">
          Estrategia Pre-Partido
        </span>
      </div>

      {/* Center: LIVE badge */}
      <div className="flex items-center gap-2">
        <span
          className="animate-live w-2 h-2 rounded-full"
          style={{ background: '#00FF87' }}
        />
        <span
          className="animate-live text-xs font-mono font-bold tracking-widest px-2 py-0.5 rounded border"
          style={{ color: '#00FF87', borderColor: '#00FF87' }}
        >
          LIVE
        </span>
      </div>

      {/* Right: Backtesting pill */}
      <div
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-mono"
        style={{ borderColor: '#3B82F6', color: '#93C5FD', background: 'rgba(59,130,246,0.08)' }}
      >
        <span>📊</span>
        <span>Precisión 30d:</span>
        <span className="font-bold" style={{ color: '#3B82F6' }}>74%</span>
        <span className="text-slate-500">|</span>
        <span>847 partidos</span>
      </div>
    </header>
  );
}
