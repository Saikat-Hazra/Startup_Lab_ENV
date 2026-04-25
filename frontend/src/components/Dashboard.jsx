import React from "react";

export default function Dashboard({ state, actions, rewards, insights, reasonings, commentary }) {
  const startups = state?.startups || [];

  return (
    <div className="space-y-4">
      {commentary && (
        <div className="rounded-xl border p-4 shadow-sm bg-gray-800 text-white">
          <h3 className="text-lg font-semibold mb-2">🎙️ Auto Commentary</h3>
          <p className="text-sm">{commentary}</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {startups.map((startup, index) => (
          <div key={index} className="rounded-xl border p-4 shadow-sm bg-gray-800 text-white">
            <h3 className="text-lg font-semibold mb-2">Startup {index + 1}</h3>
            <p className="text-sm">Cash: ${startup.cash?.toFixed?.(2) ?? startup.cash}</p>
            <p className="text-sm">Quality: {startup.product_quality?.toFixed?.(2) ?? startup.product_quality}</p>
            <p className="text-sm mt-2">Action: {actions?.[index] || "-"}</p>
            <p className="text-sm">Reward: {rewards?.[index]?.toFixed?.(2) ?? "-"}</p>
          </div>
        ))}
      </div>

      <div className="rounded-xl border p-4 shadow-sm bg-gray-800 text-white">
        <h3 className="text-lg font-semibold mb-2">Reasoning Log</h3>
        {reasonings?.length ? (
          <div className="space-y-2">
            {reasonings.map((reasoning, idx) => (
              <div key={idx} className="text-sm">
                <strong>Action:</strong> {actions?.[idx] || "-"}<br />
                <strong>Reason:</strong> {reasoning}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-600">No reasoning yet.</p>
        )}
      </div>

      <div className="rounded-xl border p-4 shadow-sm bg-gray-800 text-white">
        <h3 className="text-lg font-semibold mb-2">Failure Case</h3>
        {rewards?.some(r => r < 0) ? (
          <div className="text-sm">
            <p className="text-red-600">"Here it failed initially..."</p>
            <p>Negative reward detected. "After learning, it corrected itself."</p>
          </div>
        ) : (
          <p className="text-sm text-gray-300">No recent failures.</p>
        )}
      </div>
    </div>
  );
}
