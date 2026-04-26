"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface QValuesGraphProps {
  qValues: Record<string, number>;
  selectedAction?: string;
  handlingScore?: number;
}

const ACTION_LABELS: Record<string, string> = {
  explain: "Explain",
  correct_fact: "Correct",
  worked_example: "Example",
  analogize: "Analogy",
  question: "Query",
};

const ACTION_COLORS: Record<string, string> = {
  explain: "#6366f1",
  correct_fact: "#22d3ee",
  worked_example: "#10b981",
  analogize: "#8b5cf6",
  question: "#f59e0b",
};

export default function QValuesGraph({ qValues, selectedAction, handlingScore = 85 }: QValuesGraphProps) {
  const data = Object.entries(qValues).map(([key, value]) => ({
    name: ACTION_LABELS[key] || key,
    id: key,
    q: +value.toFixed(2),
  }));

  return (
    <div style={{ width: "100%", marginTop: "12px", background: "rgba(0,0,0,0.2)", borderRadius: "8px", padding: "10px", border: "1px solid rgba(255,255,255,0.05)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
        <div style={{ fontSize: "9px", color: "#4a5580", fontWeight: 700, letterSpacing: "0.1em" }}>
          DQN STRATEGY CONFIDENCE (Q-VALUES)
        </div>
        {selectedAction && (
          <div style={{ fontSize: "9px", color: ACTION_COLORS[selectedAction], fontWeight: 700 }}>
            SELECTED: {ACTION_LABELS[selectedAction].toUpperCase()}
          </div>
        )}
      </div>
      
      <div style={{ height: "100px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, left: 40, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#1a2a36" />
            <XAxis type="number" hide domain={['auto', 'auto']} />
            <YAxis 
              dataKey="name" 
              type="category" 
              axisLine={false} 
              tickLine={false} 
              tick={{ fill: "#8899cc", fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}
              width={60}
            />
            <Tooltip 
              cursor={{ fill: "rgba(255,255,255,0.05)" }}
              contentStyle={{ background: "#050810", border: "1px solid #1a2a36", fontSize: "10px", borderRadius: "4px" }}
            />
            <Bar dataKey="q" radius={[0, 4, 4, 0]} barSize={8}>
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={ACTION_COLORS[entry.id] || "#8899cc"} 
                  fillOpacity={entry.id === selectedAction ? 1 : 0.3}
                  stroke={entry.id === selectedAction ? "#fff" : "none"}
                  strokeWidth={entry.id === selectedAction ? 1 : 0}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Handling Score Bar */}
      <div style={{ marginTop: "12px", paddingTop: "8px", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "9px", color: "#4a5580", marginBottom: "4px", fontWeight: 700 }}>
          <span>HYBRID HANDLING SCORE</span>
          <span style={{ color: "#5bc4f5" }}>{handlingScore}%</span>
        </div>
        <div style={{ height: "4px", background: "#1a2a36", borderRadius: "2px", overflow: "hidden" }}>
          <div 
            style={{ 
              height: "100%", 
              width: `${handlingScore}%`, 
              background: "linear-gradient(90deg, #1e6fa5, #5bc4f5)",
              borderRadius: "2px",
              boxShadow: "0 0 8px rgba(91, 196, 245, 0.4)"
            }} 
          />
        </div>
      </div>
    </div>
  );
}
