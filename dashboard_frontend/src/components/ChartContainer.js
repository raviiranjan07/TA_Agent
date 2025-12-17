import { useState, useCallback } from 'react';
import { RefreshCw } from 'lucide-react';
import TradingViewChart from './TradingViewChart';
import ChartToolbar from './ChartToolbar';
import { useTradingContext } from '../context/TradingContext';

const ChartContainer = () => {
  const { chartData, isLoading, theme } = useTradingContext();
  const [selectedTool, setSelectedTool] = useState('cursor');

  const handleToolSelect = useCallback((toolId) => {
    setSelectedTool(toolId);
  }, []);

  const handleZoomIn = useCallback(() => {
    console.log('Zoom in - use mouse wheel or pinch');
  }, []);

  const handleZoomOut = useCallback(() => {
    console.log('Zoom out - use mouse wheel or pinch');
  }, []);

  const handleResetZoom = useCallback(() => {
    console.log('Reset zoom - double click on chart');
  }, []);

  const handleScreenshot = useCallback(() => {
    console.log('Screenshot - install html2canvas for this feature');
  }, []);

  const handleClearDrawings = useCallback(() => {
    console.log('Clear drawings');
  }, []);

  if (isLoading && chartData.length === 0) {
    return (
      <div className={`${theme.card}`}>
        <ChartToolbar
          selectedTool={selectedTool}
          onToolSelect={handleToolSelect}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onResetZoom={handleResetZoom}
          onScreenshot={handleScreenshot}
          onClearDrawings={handleClearDrawings}
        />
        <div className="flex items-center justify-center h-[600px]">
          <div className={`${theme.textSecondary} text-center`}>
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
            <p>Loading chart data...</p>
          </div>
        </div>
      </div>
    );
  }

  if (chartData.length === 0) {
    return (
      <div className={`${theme.card}`}>
        <ChartToolbar
          selectedTool={selectedTool}
          onToolSelect={handleToolSelect}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onResetZoom={handleResetZoom}
          onScreenshot={handleScreenshot}
          onClearDrawings={handleClearDrawings}
        />
        <div className="flex items-center justify-center h-[600px]">
          <div className={`${theme.textSecondary} text-center`}>
            <p className="text-lg font-semibold mb-2">No data available</p>
            <p className="text-sm">Make sure your backend API is running on localhost:8000</p>
            <p className="text-xs mt-2">and your database has OHLCV data</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`${theme.card}`}>
      <ChartToolbar
        selectedTool={selectedTool}
        onToolSelect={handleToolSelect}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onResetZoom={handleResetZoom}
        onScreenshot={handleScreenshot}
        onClearDrawings={handleClearDrawings}
      />
      <TradingViewChart />
    </div>
  );
};

export default ChartContainer;
