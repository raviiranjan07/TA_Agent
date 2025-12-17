import React from 'react';
import {
  MousePointer2,
  TrendingUp,
  Minus,
  Square,
  Type,
  Trash2,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Camera,
} from 'lucide-react';
import { useTradingContext } from '../context/TradingContext';

const ChartToolbar = ({
  onToolSelect,
  selectedTool,
  onZoomIn,
  onZoomOut,
  onResetZoom,
  onScreenshot,
  onClearDrawings
}) => {
  const { theme, isDarkMode } = useTradingContext();

  const tools = [
    { id: 'cursor', icon: MousePointer2, label: 'Cursor' },
    { id: 'trendline', icon: TrendingUp, label: 'Trend Line' },
    { id: 'horizontal', icon: Minus, label: 'Horizontal Line' },
    { id: 'rectangle', icon: Square, label: 'Rectangle' },
    { id: 'text', icon: Type, label: 'Text' },
  ];

  const actions = [
    { id: 'zoomIn', icon: ZoomIn, label: 'Zoom In', onClick: onZoomIn },
    { id: 'zoomOut', icon: ZoomOut, label: 'Zoom Out', onClick: onZoomOut },
    { id: 'reset', icon: Maximize2, label: 'Reset Zoom', onClick: onResetZoom },
    { id: 'screenshot', icon: Camera, label: 'Screenshot', onClick: onScreenshot },
    { id: 'clear', icon: Trash2, label: 'Clear Drawings', onClick: onClearDrawings },
  ];

  return (
    <div className={`flex items-center gap-1 p-1 ${theme.card} border-b ${theme.border}`}>
      {/* Drawing Tools */}
      <div className="flex items-center gap-1 border-r border-gray-600 pr-2 mr-2">
        {tools.map((tool) => (
          <button
            key={tool.id}
            onClick={() => onToolSelect(tool.id)}
            className={`p-2 rounded transition-colors ${
              selectedTool === tool.id
                ? 'bg-blue-600 text-white'
                : `${theme.hover} ${theme.textSecondary}`
            }`}
            title={tool.label}
          >
            <tool.icon className="w-4 h-4" />
          </button>
        ))}
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-1">
        {actions.map((action) => (
          <button
            key={action.id}
            onClick={action.onClick}
            className={`p-2 rounded transition-colors ${theme.hover} ${theme.textSecondary}`}
            title={action.label}
          >
            <action.icon className="w-4 h-4" />
          </button>
        ))}
      </div>
    </div>
  );
};

export default ChartToolbar;
