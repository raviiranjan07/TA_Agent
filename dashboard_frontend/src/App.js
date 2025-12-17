import { TradingProvider, useTradingContext } from './context/TradingContext';
import { useTradingData, useWebSocket } from './hooks';
import {
  Header,
  TimeframeBar,
  StatsBar,
  ChartContainer,
  Footer,
  ErrorBoundary,
} from './components';

const DashboardContent = () => {
  const { theme, isLoading } = useTradingContext();
  const { fetchData } = useTradingData();
  useWebSocket();

  return (
    <div className={`min-h-screen ${theme.bg} transition-colors duration-300`}>
      <Header onRefresh={fetchData} isLoading={isLoading} />
      <TimeframeBar />
      <StatsBar />
      <ChartContainer />
      <Footer />
    </div>
  );
};

const TradingViewDashboard = () => {
  return (
    <ErrorBoundary>
      <TradingProvider>
        <DashboardContent />
      </TradingProvider>
    </ErrorBoundary>
  );
};

export default TradingViewDashboard;
