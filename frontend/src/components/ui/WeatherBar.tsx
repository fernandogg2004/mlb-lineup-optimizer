import { classifyERA } from '../../utils/statLabels';

interface WeatherBarProps {
  venue: string;
  pitcher: string;
  pitcherHand: string;
  pitcherERA: number;
}

export default function WeatherBar({
  venue,
  pitcher,
  pitcherHand,
  pitcherERA,
}: WeatherBarProps) {
  return (
    <div
      className="flex items-center gap-4 px-6 h-10 border-b overflow-x-auto"
      style={{ background: '#0D1117', borderColor: '#1F2937' }}
    >
      {/* Wind */}
      <Pill>
        <span>🌬️</span>
        <span>Viento: <span className="text-white">12mph OUT</span></span>
      </Pill>

      {/* Humidity */}
      <Pill>
        <span>💧</span>
        <span>Humedad: <span className="text-white">67%</span></span>
      </Pill>

      {/* Park Factor */}
      <Pill>
        <span>🏟️</span>
        <span>Park Factor: <span className="text-white">0.94</span></span>
        <span
          className="ml-1 px-1.5 py-0.5 rounded text-xs font-mono font-bold"
          style={{ color: '#FF4560', background: 'rgba(255,69,96,0.12)' }}
        >
          E[R] -0.22
        </span>
      </Pill>

      {/* Divider */}
      <span className="text-slate-700">|</span>

      {/* Venue + Pitcher */}
      <span className="text-xs font-mono text-slate-400 whitespace-nowrap">
        <span className="text-slate-300">{venue}</span>
        <span className="text-slate-600"> · </span>
        <span className="text-white font-semibold">{pitcher}</span>
        <span className="text-slate-400"> ({pitcherHand}HP)</span>
        <span className="text-slate-600"> · </span>
        <span>ERA </span>
        <span
          className="font-bold"
          style={{ color: classifyERA(pitcherERA).color }}
        >
          {pitcherERA.toFixed(2)}
        </span>
        {pitcherERA > 0 && (
          <span
            className="ml-1 px-1.5 py-0.5 rounded text-[10px] font-mono font-bold"
            style={{
              color: classifyERA(pitcherERA).color,
              background: classifyERA(pitcherERA).bgColor,
            }}
          >
            {classifyERA(pitcherERA).text}
          </span>
        )}
      </span>
    </div>
  );
}

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-mono text-slate-400 whitespace-nowrap border"
      style={{ background: '#1a2035', borderColor: '#374151' }}
    >
      {children}
    </span>
  );
}
