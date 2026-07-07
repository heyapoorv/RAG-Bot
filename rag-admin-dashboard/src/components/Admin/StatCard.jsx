export default function StatCard({ label, value, icon }) {
  return (
    <div className="stat-card">
      <div className="flex items-center justify-center mb-4">
        <div className="text-3xl">{icon}</div>
      </div>
      <div className="text-4xl font-bold mb-2 text-primary">{value}</div>
      <div className="text-sm text-secondary font-medium">{label}</div>
    </div>
  );
}