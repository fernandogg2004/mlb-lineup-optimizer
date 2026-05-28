/** Calibrated MLB metric thresholds (Bloque 2 — dynamic percentile labels). */

export interface MetricLabel {
  text: string;
  color: string;
  bgColor: string;
}

export function classifyISO(iso: number): MetricLabel {
  if (iso < 0.080) return { text: 'Poder Bajo',          color: '#FF4560', bgColor: 'rgba(255,69,96,0.12)' };
  if (iso < 0.140) return { text: 'Poder Moderado',      color: '#FFB800', bgColor: 'rgba(255,184,0,0.12)' };
  if (iso < 0.200) return { text: 'Buen Poder',          color: '#3B82F6', bgColor: 'rgba(59,130,246,0.12)' };
  if (iso < 0.250) return { text: 'Poder Sobresaliente', color: '#00FF87', bgColor: 'rgba(0,255,135,0.12)' };
  return             { text: 'Poder Élite',             color: '#00FF87', bgColor: 'rgba(0,255,135,0.20)' };
}

export function classifyWOBA(woba: number): MetricLabel {
  if (woba < 0.290) return { text: 'Bajo',              color: '#FF4560', bgColor: 'rgba(255,69,96,0.12)' };
  if (woba < 0.320) return { text: 'Promedio',          color: '#FFB800', bgColor: 'rgba(255,184,0,0.12)' };
  if (woba < 0.360) return { text: 'Sobre el Promedio', color: '#3B82F6', bgColor: 'rgba(59,130,246,0.12)' };
  if (woba < 0.400) return { text: 'Excelente',         color: '#00FF87', bgColor: 'rgba(0,255,135,0.12)' };
  return              { text: 'Élite',                 color: '#00FF87', bgColor: 'rgba(0,255,135,0.20)' };
}

export function classifyOBP(obp: number): MetricLabel {
  if (obp < 0.290) return { text: 'Bajo',          color: '#FF4560', bgColor: 'rgba(255,69,96,0.12)' };
  if (obp < 0.320) return { text: 'Promedio',      color: '#FFB800', bgColor: 'rgba(255,184,0,0.12)' };
  if (obp < 0.350) return { text: 'Bueno',         color: '#3B82F6', bgColor: 'rgba(59,130,246,0.12)' };
  if (obp < 0.390) return { text: 'Sobresaliente', color: '#00FF87', bgColor: 'rgba(0,255,135,0.12)' };
  return             { text: 'Élite',             color: '#00FF87', bgColor: 'rgba(0,255,135,0.20)' };
}

export function classifyOPS(ops: number): MetricLabel {
  if (ops < 0.650) return { text: 'Bajo',              color: '#FF4560', bgColor: 'rgba(255,69,96,0.12)' };
  if (ops < 0.750) return { text: 'Bajo del Promedio', color: '#FF8C42', bgColor: 'rgba(255,140,66,0.12)' };
  if (ops < 0.850) return { text: 'Promedio-Alto',     color: '#FFB800', bgColor: 'rgba(255,184,0,0.12)' };
  if (ops < 0.950) return { text: 'Bueno',             color: '#3B82F6', bgColor: 'rgba(59,130,246,0.12)' };
  return             { text: 'Élite',                 color: '#00FF87', bgColor: 'rgba(0,255,135,0.20)' };
}

/**
 * ERA classification from the BATTER'S perspective.
 * High ERA = pitcher is hittable = advantage for the batting team.
 */
export function classifyERA(era: number): MetricLabel {
  if (!era || era <= 0)
    return { text: 'ERA N/D',              color: '#6B7280', bgColor: 'rgba(107,114,128,0.12)' };
  if (era < 3.00)
    return { text: 'Amenaza Alta',         color: '#FF4560', bgColor: 'rgba(255,69,96,0.12)' };
  if (era < 4.00)
    return { text: 'Amenaza Moderada',     color: '#FFB800', bgColor: 'rgba(255,184,0,0.12)' };
  if (era < 4.75)
    return { text: 'Favorable',            color: '#3B82F6', bgColor: 'rgba(59,130,246,0.12)' };
  return   { text: 'Ventaja Ofensiva Clara', color: '#00FF87', bgColor: 'rgba(0,255,135,0.12)' };
}
