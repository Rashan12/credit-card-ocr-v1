import { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface Metric {
  id: number;
  timestamp: string;
  side: string;
  vision_success: number;
  vit_accuracy_card: number;
  vit_accuracy_expiry: number;
  vit_accuracy_security: number;
  vision_confidence_card: number;
  vision_confidence_expiry: number;
  vision_confidence_security: number;
  vit_confidence_card: number;
  vit_confidence_expiry: number;
  vit_confidence_security: number;
  user_correction: number;
  used_vit_card: number;
  used_vit_expiry: number;
  used_vit_security: number;
}

interface TrainingLog {
  id: number;
  timestamp: string;
  step: number;
  epoch: number;
  total_loss: number;
  card_loss: number;
  expiry_loss: number;
  security_loss: number;
  learning_rate: number;
}

interface TransitionLog {
  id: number;
  timestamp: string;
  field: string;
  used_vit: number;
  fallback_triggered: number;
  vit_accuracy: number;
  vision_accuracy: number;
}

interface AccuracyTrend {
  timestamp: string;
  vit_accuracy: number;
  vision_accuracy: number;
}

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [trainingLogs, setTrainingLogs] = useState<TrainingLog[]>([]);
  const [transitionLogs, setTransitionLogs] = useState<TransitionLog[]>([]);
  const [accuracyTrends, setAccuracyTrends] = useState<{
    card_number: AccuracyTrend[];
    expiry_date: AccuracyTrend[];
    security_number: AccuracyTrend[];
  }>({ card_number: [], expiry_date: [], security_number: [] });

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('http://localhost:5000/metrics', { mode: 'cors' });
        const data = await response.json();
        setMetrics(data.metrics);
        setTrainingLogs(data.training_logs);
        setTransitionLogs(data.transition_logs);
        setAccuracyTrends(data.accuracy_trends);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  // Summary Metrics
  const visionSuccessRate = metrics.length > 0
    ? (metrics.filter(m => m.vision_success === 1).length / metrics.length) * 100
    : 0;
  const userCorrectionRate = metrics.length > 0
    ? (metrics.filter(m => m.user_correction === 1).length / metrics.length) * 100
    : 0;
  const avgVitAccuracyCard = metrics.filter(m => m.side === 'front').length > 0
    ? metrics.filter(m => m.side === 'front').reduce((sum, m) => sum + m.vit_accuracy_card, 0) / metrics.filter(m => m.side === 'front').length * 100
    : 0;
  const avgVitAccuracyExpiry = metrics.filter(m => m.side === 'front').length > 0
    ? metrics.filter(m => m.side === 'front').reduce((sum, m) => sum + m.vit_accuracy_expiry, 0) / metrics.filter(m => m.side === 'front').length * 100
    : 0;
  const avgVitAccuracySecurity = metrics.filter(m => m.side === 'back').length > 0
    ? metrics.filter(m => m.side === 'back').reduce((sum, m) => sum + m.vit_accuracy_security, 0) / metrics.filter(m => m.side === 'back').length * 100
    : 0;

  // Transition Readiness
  const isReadyForTransition = (field: string, trends: AccuracyTrend[]) => {
    if (trends.length < 14) return false; // Need at least 14 days of data
    const recentTrends = trends.slice(-14);
    return recentTrends.every(trend => trend.vit_accuracy > trend.vision_accuracy);
  };

  // Chart Data
  const accuracyCardData = {
    labels: accuracyTrends.card_number.map(trend => trend.timestamp),
    datasets: [
      {
        label: 'ViT Card Number Accuracy',
        data: accuracyTrends.card_number.map(trend => trend.vit_accuracy * 100),
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: false,
      },
      {
        label: 'Vision API Card Number Accuracy',
        data: accuracyTrends.card_number.map(trend => trend.vision_accuracy * 100),
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: false,
      },
    ],
  };

  const accuracyExpiryData = {
    labels: accuracyTrends.expiry_date.map(trend => trend.timestamp),
    datasets: [
      {
        label: 'ViT Expiry Date Accuracy',
        data: accuracyTrends.expiry_date.map(trend => trend.vit_accuracy * 100),
        borderColor: 'rgba(153, 102, 255, 1)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        fill: false,
      },
      {
        label: 'Vision API Expiry Date Accuracy',
        data: accuracyTrends.expiry_date.map(trend => trend.vision_accuracy * 100),
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: false,
      },
    ],
  };

  const accuracySecurityData = {
    labels: accuracyTrends.security_number.map(trend => trend.timestamp),
    datasets: [
      {
        label: 'ViT Security Number Accuracy',
        data: accuracyTrends.security_number.map(trend => trend.vit_accuracy * 100),
        borderColor: 'rgba(255, 159, 64, 1)',
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        fill: false,
      },
      {
        label: 'Vision API Security Number Accuracy',
        data: accuracyTrends.security_number.map(trend => trend.vision_accuracy * 100),
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: false,
      },
    ],
  };

  const lossData = {
    labels: trainingLogs.map(log => log.timestamp),
    datasets: [
      {
        label: 'Total Loss',
        data: trainingLogs.map(log => log.total_loss),
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: false,
      },
      {
        label: 'Card Loss',
        data: trainingLogs.map(log => log.card_loss),
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        fill: false,
      },
      {
        label: 'Expiry Loss',
        data: trainingLogs.map(log => log.expiry_loss),
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: false,
      },
      {
        label: 'Security Loss',
        data: trainingLogs.map(log => log.security_loss),
        borderColor: 'rgba(153, 102, 255, 1)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        fill: false,
      },
    ],
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-indigo-800 text-center">
        Monitoring Dashboard
      </h1>

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700">Vision API Success Rate</h2>
          <p className="text-2xl text-gray-700">{visionSuccessRate.toFixed(2)}%</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700">User Correction Rate</h2>
          <p className="text-2xl text-gray-700">{userCorrectionRate.toFixed(2)}%</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700">Average ViT Card Accuracy</h2>
          <p className="text-2xl text-gray-700">{avgVitAccuracyCard.toFixed(2)}%</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700">Average ViT Expiry Accuracy</h2>
          <p className="text-2xl text-gray-700">{avgVitAccuracyExpiry.toFixed(2)}%</p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700">Average ViT Security Accuracy</h2>
          <p className="text-2xl text-gray-700">{avgVitAccuracySecurity.toFixed(2)}%</p>
        </div>
      </div>

      {/* Transition Readiness */}
      <div className="bg-white p-4 rounded-lg shadow-lg mb-6">
        <h2 className="text-xl font-semibold text-indigo-700 mb-4">Transition Readiness</h2>
        <div className="space-y-2">
          <p>
            <strong>Card Number:</strong> {isReadyForTransition("card_number", accuracyTrends.card_number) ? "Ready" : "Not Ready"}
            {isReadyForTransition("card_number", accuracyTrends.card_number) ? " (ViT consistently better for 14 days)" : ""}
          </p>
          <p>
            <strong>Expiry Date:</strong> {isReadyForTransition("expiry_date", accuracyTrends.expiry_date) ? "Ready" : "Not Ready"}
            {isReadyForTransition("expiry_date", accuracyTrends.expiry_date) ? " (ViT consistently better for 14 days)" : ""}
          </p>
          <p>
            <strong>Security Number:</strong> {isReadyForTransition("security_number", accuracyTrends.security_number) ? "Ready" : "Not Ready"}
            {isReadyForTransition("security_number", accuracyTrends.security_number) ? " (ViT consistently better for 14 days)" : ""}
          </p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700 mb-4">Card Number Accuracy Over Time</h2>
          <Line data={accuracyCardData} options={{ responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }} />
        </div>
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700 mb-4">Expiry Date Accuracy Over Time</h2>
          <Line data={accuracyExpiryData} options={{ responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }} />
        </div>
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700 mb-4">Security Number Accuracy Over Time</h2>
          <Line data={accuracySecurityData} options={{ responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }} />
        </div>
        <div className="bg-white p-4 rounded-lg shadow-lg">
          <h2 className="text-xl font-semibold text-indigo-700 mb-4">Training Loss Over Time</h2>
          <Line data={lossData} options={{ responsive: true, scales: { y: { beginAtZero: true } } }} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;