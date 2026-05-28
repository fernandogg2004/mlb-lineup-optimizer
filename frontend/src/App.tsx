import { useState } from 'react';
import { GameProvider } from './context/GameContext';
import WarRoom from './pages/WarRoom';
import PostGame from './pages/PostGame';
import TrackRecord from './pages/TrackRecord';

type Tab = 'war_room' | 'post_game' | 'track_record';

const TABS: { id: Tab; label: string }[] = [
  { id: 'war_room',     label: '🏟️  War Room' },
  { id: 'post_game',   label: '📊  Post-Game Review' },
  { id: 'track_record', label: '📋  Track Record' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('war_room');

  return (
    <GameProvider>
      {/* Global tab bar */}
      <div
        style={{
          background: '#060C18',
          borderBottom: '1px solid #182840',
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          padding: '0 16px',
          position: 'sticky',
          top: 0,
          zIndex: 50,
        }}
      >
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '12px 18px',
              fontSize: '0.82rem',
              fontWeight: activeTab === tab.id ? 700 : 500,
              color: activeTab === tab.id ? '#90CAF9' : '#3D6080',
              background: 'none',
              border: 'none',
              borderBottom: activeTab === tab.id
                ? '2px solid #3B82F6'
                : '2px solid transparent',
              cursor: 'pointer',
              letterSpacing: '0.02em',
              transition: 'all 0.15s ease',
              marginBottom: -1,
            }}
          >
            {tab.label}
          </button>
        ))}

        <div style={{ marginLeft: 'auto', fontSize: '0.63rem', color: '#1E3850', fontFamily: 'JetBrains Mono, monospace' }}>
          MLB War Room v2.1
        </div>
      </div>

      {/* Page content — GameProvider wraps all so context persists across tabs */}
      <div style={{ display: activeTab === 'war_room' ? 'block' : 'none' }}>
        <WarRoom />
      </div>
      <div style={{ display: activeTab === 'post_game' ? 'block' : 'none' }}>
        <PostGame />
      </div>
      <div style={{ display: activeTab === 'track_record' ? 'block' : 'none' }}>
        <TrackRecord />
      </div>
    </GameProvider>
  );
}
