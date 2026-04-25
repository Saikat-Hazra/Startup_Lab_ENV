import React from "react";

export default function Dashboard({ state, actions, rewards, insights }) {
  const startups = state?.startups || [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {startups.map((startup, index) => (
          <div key={index} className="rounded-xl border p-4 shadow-sm bg-white">
            <h3 className="text-lg font-semibold mb-2">Startup {index + 1}</h3>
            <p className="text-sm">Cash: ${startup.cash?.toFixed?.(2) ?? startup.cash}</p>
            <p className="text-sm">Quality: {startup.product_quality?.toFixed?.(2) ?? startup.product_quality}</p>
            <p className="text-sm mt-2">Action: {actions?.[index] || "-"}</p>
            <p className="text-sm">Reward: {rewards?.[index]?.toFixed?.(2) ?? "-"}</p>
          </div>
        ))}
      </div>

      <div className="rounded-xl border p-4 shadow-sm bg-white">
        <h3 className="text-lg font-semibold mb-2">Insights</h3>
        {insights?.length ? (
          <ul className="list-disc pl-5 space-y-1 text-sm">
            {insights.map((insight, idx) => (
              <li key={idx}>{insight}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-gray-600">No insights yet.</p>
        )}
      </div>
    </div>
  );
}
