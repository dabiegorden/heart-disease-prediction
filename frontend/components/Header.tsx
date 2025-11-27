export default function Header() {
  return (
    <header className="border-b border-slate-800/50 bg-slate-950/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-transparent bg-linear-to-r from-blue-400 via-cyan-400 to-blue-500 bg-clip-text">
            CardioPredict AI
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Clinical Heart Disease Risk Assessment
          </p>
        </div>
        <div className="text-right text-sm text-slate-400">
          <p>Powered by 6 ML Models</p>
          <p className="text-xs mt-1">Real-time Predictions</p>
        </div>
      </div>
    </header>
  );
}
