import { useTradingContext } from '../context/TradingContext';

const Footer = () => {
  const { chartData, selectedPair, selectedTimeframe, theme } = useTradingContext();

  // Get date range from chart data
  const getDateRange = () => {
    if (!chartData || chartData.length === 0) return '';

    const firstCandle = chartData[0];
    const lastCandle = chartData[chartData.length - 1];

    const formatDate = (timeValue) => {
      try {
        let date;
        if (typeof timeValue === 'string') {
          date = new Date(timeValue);
        } else if (timeValue > 1e12) {
          date = new Date(timeValue);
        } else {
          date = new Date(timeValue * 1000);
        }

        return date.toLocaleString('en-IN', {
          timeZone: 'Asia/Kolkata',
          day: '2-digit',
          month: 'short',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          hour12: false,
        });
      } catch {
        return 'N/A';
      }
    };

    const startDate = formatDate(firstCandle.time);
    const endDate = formatDate(lastCandle.time);

    return `${startDate} → ${endDate}`;
  };

  // Current time in IST
  const getCurrentTime = () => {
    return new Date().toLocaleString('en-IN', {
      timeZone: 'Asia/Kolkata',
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  return (
    <div className={`${theme.card} border-t ${theme.border} px-4 py-2`}>
      <div className="flex items-center justify-between text-xs">
        <div className={`${theme.textSecondary} flex items-center gap-4`}>
          <span>
            <strong className={theme.text}>{selectedPair}</strong> • {selectedTimeframe.toUpperCase()}
          </span>
          <span className="hidden md:inline">
            IST (Asia/Kolkata)
          </span>
          <span className="hidden lg:inline">
            {getCurrentTime()}
          </span>
        </div>

        <div className={`${theme.textSecondary} flex items-center gap-4`}>
          {chartData.length > 0 && (
            <>
              <span className="hidden md:inline">
                {getDateRange()}
              </span>
              <span>
                <strong className={theme.text}>{chartData.length}</strong> candles
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Footer;
