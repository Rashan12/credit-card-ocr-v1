import { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import { FaCreditCard, FaCalendarAlt, FaCamera, FaUpload, FaLock, FaSpinner, FaCheckCircle } from 'react-icons/fa';
import './App.css';

const App: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const [showWebcam, setShowWebcam] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<'upload' | 'processing' | 'results' | 'feedback' | 'confirmation' | 'proceed'>('upload');
  const [frontDetails, setFrontDetails] = useState<{
    final: { card_number: string; expiry_date: string; card_type: string; errors?: string[] };
    confidence_scores: { card_number: number; expiry_date: number; security_number: number };
    used_vit: { card_number: boolean; expiry_date: boolean; security_number: boolean };
  }>({ 
    final: { card_number: '', expiry_date: '', card_type: '' },
    confidence_scores: { card_number: 0, expiry_date: 0, security_number: 0 },
    used_vit: { card_number: false, expiry_date: false, security_number: false },
  });
  const [backDetails, setBackDetails] = useState<{
    final: { security_number: string; errors?: string[] };
    confidence_scores: { card_number: number; expiry_date: number; security_number: number };
    used_vit: { card_number: boolean; expiry_date: boolean; security_number: boolean };
  }>({ 
    final: { security_number: '' },
    confidence_scores: { card_number: 0, expiry_date: 0, security_number: 0 },
    used_vit: { card_number: false, expiry_date: false, security_number: false },
  });
  const [capturedImages, setCapturedImages] = useState<{
    front: string | null;
    back: string | null;
  }>({ front: null, back: null });
  const [isCapturingFront, setIsCapturingFront] = useState(true);
  const [uploadedFiles, setUploadedFiles] = useState<{
    front: File | null;
    back: File | null;
  }>({ front: null, back: null });
  const [uploadedImages, setUploadedImages] = useState<{
    front: string | null;
    back: string | null;
  }>({ front: null, back: null });
  const [feedback, setFeedback] = useState<{
    card_number: string;
    expiry_date: string;
    security_number: string;
  }>({ card_number: '', expiry_date: '', security_number: '' });
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackSide, setFeedbackSide] = useState<'front' | 'back'>('front');
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [confirmationData, setConfirmationData] = useState<any>(null);
  const [totalProcessingTime, setTotalProcessingTime] = useState<number>(0);
  const [isLiveScanning, setIsLiveScanning] = useState(false);
  const [liveOcrResult, setLiveOcrResult] = useState<any>(null);
  const [instruction, setInstruction] = useState<string>("");

  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    if (isLiveScanning && showWebcam) {
      intervalId = setInterval(async () => {
        const imageSrc = webcamRef.current?.getScreenshot();
        if (imageSrc) {
          const response = await fetch(imageSrc);
          const blob = await response.blob();
          const formData = new FormData();
          formData.append('image', blob, 'frame.jpg');
          formData.append('is_front', isCapturingFront ? 'true' : 'false');
          try {
            const res = await fetch('http://localhost:5000/live_ocr', {
              method: 'POST',
              body: formData,
            });
            const data = await res.json();
            setLiveOcrResult(data.ocr_result);
            setInstruction(data.instruction);
          } catch (error) {
            console.error('Error in live OCR:', error);
          }
        }
      }, 500); // Every 500ms
    }
    return () => clearInterval(intervalId);
  }, [isLiveScanning, showWebcam, isCapturingFront]);

  const captureImage = async () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      if (isCapturingFront) {
        setCapturedImages({ ...capturedImages, front: imageSrc });
        setIsCapturingFront(false);
      } else {
        setCapturedImages({ ...capturedImages, back: imageSrc });
        setShowWebcam(false);
        setIsLiveScanning(false);
        await processImages();
      }
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>, side: 'front' | 'back') => {
    const file = event.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      if (side === 'front') {
        setUploadedFiles({ ...uploadedFiles, front: file });
        setUploadedImages({ ...uploadedImages, front: imageUrl });
      } else {
        setUploadedFiles({ ...uploadedFiles, back: file });
        setUploadedImages({ ...uploadedImages, back: imageUrl });
      }
    }
  };

  const processImages = async () => {
    if (!uploadedFiles.front || !uploadedFiles.back) {
      alert("Please upload both front and back images before scanning.");
      return;
    }

    setIsLoading(true);
    setCurrentStep('processing');

    const formData = new FormData();
    formData.append('front_image', uploadedFiles.front);
    formData.append('back_image', uploadedFiles.back);

    try {
      const response = await fetch('http://localhost:5000/ocr', {
        method: 'POST',
        body: formData,
        mode: 'cors',
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} - ${await response.text()}`);
      }
      const data = await response.json();
      console.log('Response data:', data);
      if (data.error) {
        setFrontDetails({ 
          final: { card_number: 'Error', expiry_date: 'Error', card_type: '', errors: [data.error] },
          confidence_scores: { card_number: 0, expiry_date: 0, security_number: 0 },
          used_vit: { card_number: false, expiry_date: false, security_number: false },
        });
        setBackDetails({
          final: { security_number: 'Error', errors: [data.error] },
          confidence_scores: { card_number: 0, expiry_date: 0, security_number: 0 },
          used_vit: { card_number: false, expiry_date: false, security_number: false },
        });
      } else {
        setFrontDetails({
          final: {
            card_number: data.front.card_number,
            expiry_date: data.front.expiry_date,
            card_type: data.front.card_type,
            errors: data.front.errors
          },
          confidence_scores: data.confidence_scores.front,
          used_vit: data.used_vit.front,
        });
        setBackDetails({
          final: {
            security_number: data.back.security_number,
            errors: data.back.errors
          },
          confidence_scores: data.confidence_scores.back,
          used_vit: data.used_vit.back,
        });
        setFeedback({
          card_number: data.front.card_number || '',
          expiry_date: data.front.expiry_date || '',
          security_number: data.back.security_number || ''
        });
        setTotalProcessingTime(data.processing_time);
      }
      setCurrentStep('results');
    } catch (error) {
      console.error('Error processing images:', error);
      setFrontDetails({ 
        final: { card_number: 'Error', expiry_date: 'Error', card_type: '', errors: ['Failed to process images: ' + (error as Error).message] },
        confidence_scores: { card_number: 0, expiry_date: 0, security_number: 0 },
        used_vit: { card_number: false, expiry_date: false, security_number: false },
      });
      setBackDetails({ 
        final: { security_number: 'Error', errors: ['Failed to process images: ' + (error as Error).message] },
        confidence_scores: { card_number: 0, expiry_date: 0, security_number: 0 },
        used_vit: { card_number: false, expiry_date: false, security_number: false },
      });
      setCurrentStep('results');
    } finally {
      setIsLoading(false);
    }
  };

  const submitFeedback = async () => {
    setCurrentStep('feedback');
    try {
      const response = await fetch('http://localhost:5000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: `temp_${feedbackSide}.jpg`,
          corrections: feedback,
          side: feedbackSide
        }),
        mode: 'cors',
      });
      const result = await response.json();
      if (result.error) {
        console.error('Feedback error:', result.error);
      } else {
        setConfirmationData(result);
        setShowFeedback(false);
        setShowConfirmation(true);
        setCurrentStep('confirmation');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setCurrentStep('results');
    }
  };

  const confirmFeedback = async () => {
    try {
      const response = await fetch('http://localhost:5000/confirm_feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: confirmationData.image_path,
          corrections: confirmationData.corrected_result,
          side: confirmationData.side
        }),
        mode: 'cors',
      });
      const result = await response.json();
      if (result.error) {
        console.error('Confirmation error:', result.error);
      } else {
        if (confirmationData.side === 'front') {
          setFrontDetails({
            final: {
              card_number: confirmationData.corrected_result.card_number,
              expiry_date: confirmationData.corrected_result.expiry_date,
              card_type: frontDetails.final.card_type,
              errors: confirmationData.corrected_result.errors || []
            },
            confidence_scores: frontDetails.confidence_scores,
            used_vit: frontDetails.used_vit,
          });
        } else {
          setBackDetails({
            final: {
              security_number: confirmationData.corrected_result.security_number,
              errors: confirmationData.corrected_result.errors || []
            },
            confidence_scores: backDetails.confidence_scores,
            used_vit: backDetails.used_vit,
          });
        }
      }
      setShowConfirmation(false);
      setCurrentStep('results');
    } catch (error) {
      console.error('Error confirming feedback:', error);
      setCurrentStep('results');
    }
  };

  const handleProceed = async () => {
    try {
      const response = await fetch('http://localhost:5000/proceed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
        mode: 'cors',
      });
      const result = await response.json();
      console.log(result.message);
      // Clear local state
      setFrontDetails({
        final: { card_number: '', expiry_date: '', card_type: '' },
        confidence_scores: { card_number: 0, expiry_date: 0, security_number: 0 },
        used_vit: { card_number: false, expiry_date: false, security_number: false },
      });
      setBackDetails({
        final: { security_number: '' },
        confidence_scores: { card_number: 0, expiry_date: 0, security_number: 0 },
        used_vit: { card_number: false, expiry_date: false, security_number: false },
      });
      setCapturedImages({ front: null, back: null });
      setUploadedImages({ front: null, back: null });
      setUploadedFiles({ front: null, back: null });
      setFeedback({ card_number: '', expiry_date: '', security_number: '' });
      setTotalProcessingTime(0);
      setCurrentStep('proceed');
    } catch (error) {
      console.error('Error proceeding:', error);
    }
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100">
      <div className="scanner-container max-w-2xl w-full p-6">
        <head>
          {/* Tailwind CSS CDN */}
          <script src="https://cdn.tailwindcss.com"></script>
          {/* Google Fonts (Inter) */}
          <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
          {/* Font Awesome CDN */}
          <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
        </head>

        {/* Header */}
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-900 flex items-center justify-center bg-gradient-to-r from-indigo-600 to-indigo-800 text-white py-4 rounded-lg shadow-lg">
            <FaCreditCard className="text-yellow-300 mr-2" />
            Credit Card Scanner
          </h1>
          <p className="text-gray-600 mt-2 text-lg">Scan and extract credit card details securely and efficiently</p>
        </header>

        {/* Progress Indicator */}
        <div className="mb-8">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span className={currentStep === 'upload' ? 'text-indigo-600 font-semibold' : ''}>1. Upload</span>
            <span className={currentStep === 'processing' ? 'text-indigo-600 font-semibold' : ''}>2. Processing</span>
            <span className={currentStep === 'results' ? 'text-indigo-600 font-semibold' : ''}>3. Results</span>
            {currentStep === 'feedback' || currentStep === 'confirmation' || currentStep === 'proceed' ? (
              <span className="text-indigo-600 font-semibold">
                {currentStep === 'feedback' ? '4. Feedback' : currentStep === 'confirmation' ? '5. Confirmation' : '6. Proceed'}
              </span>
            ) : null}
          </div>
          <div className="progress-bar h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="progress-bar-fill h-full bg-gradient-to-r from-indigo-500 to-indigo-700 transition-all duration-300"
              style={{
                width: currentStep === 'upload' ? '20%' :
                       currentStep === 'processing' ? '40%' :
                       currentStep === 'results' ? '60%' :
                       currentStep === 'feedback' ? '80%' :
                       currentStep === 'confirmation' ? '90%' : '100%'
              }}
            ></div>
          </div>
        </div>

        {/* Buttons for Camera and Upload */}
        {currentStep === 'upload' && (
          <div className="flex flex-col items-center space-y-6 mb-8">
            <button
              onClick={() => {
                setShowWebcam(true);
                setIsCapturingFront(true);
                setCapturedImages({ front: null, back: null });
                setIsLiveScanning(true);
              }}
              className="btn-primary w-full sm:w-auto px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-800 text-white rounded-lg shadow-md hover:from-indigo-700 hover:to-indigo-900 transition-all duration-200 flex items-center justify-center"
            >
              <FaCamera className="mr-2" />
              Scan with Camera
            </button>

            <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4 w-full">
              <label className="btn-secondary w-full sm:w-auto cursor-pointer px-6 py-3 bg-white text-indigo-600 border-2 border-indigo-600 rounded-lg shadow-md hover:bg-indigo-50 transition-all duration-200 flex items-center justify-center">
                <FaUpload className="mr-2" />
                Upload Front
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleFileUpload(e, 'front')}
                  className="hidden"
                />
              </label>
              <label className="btn-secondary w-full sm:w-auto cursor-pointer px-6 py-3 bg-white text-indigo-600 border-2 border-indigo-600 rounded-lg shadow-md hover:bg-indigo-50 transition-all duration-200 flex items-center justify-center">
                <FaUpload className="mr-2" />
                Upload Back
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleFileUpload(e, 'back')}
                  className="hidden"
                />
              </label>
            </div>
            {(uploadedImages.front && uploadedImages.back) && (
              <button
                onClick={processImages}
                className="btn-primary w-full sm:w-auto px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-800 text-white rounded-lg shadow-md hover:from-indigo-700 hover:to-indigo-900 transition-all duration-200"
              >
                Scan Card
              </button>
            )}
          </div>
        )}

        {/* Loading Indicator */}
        {isLoading && (
          <div className="flex items-center justify-center mb-8">
            <FaSpinner className="animate-spin text-indigo-600 text-3xl mr-2" />
            <p className="text-indigo-600 text-lg font-medium">Processing your images...</p>
          </div>
        )}

        {/* Webcam Feed */}
        {showWebcam && (
          <div className="mb-8">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="webcam-feed w-full max-w-md mx-auto rounded-lg overflow-hidden shadow-lg"
              videoConstraints={{
                facingMode: 'environment',
              }}
            />
            <div className="live-feedback mt-4 text-center bg-white p-4 rounded-lg shadow-md">
              <p className="instruction text-lg font-semibold text-indigo-700">{instruction}</p>
              {liveOcrResult && (
                <div className="mt-2 text-gray-800">
                  {isCapturingFront ? (
                    <>
                      <p>Card Number: {liveOcrResult.card_number || 'Detecting...'}</p>
                      <p>Expiry Date: {liveOcrResult.expiry_date || 'Detecting...'}</p>
                    </>
                  ) : (
                    <p>Security Number: {liveOcrResult.security_number || 'Detecting...'}</p>
                  )}
                </div>
              )}
            </div>
            <div className="flex justify-center mt-4">
              <button
                onClick={captureImage}
                className="btn-primary w-full sm:w-auto px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-800 text-white rounded-lg shadow-md hover:from-indigo-700 hover:to-indigo-900 transition-all duration-200"
              >
                Capture {isCapturingFront ? 'Front' : 'Back'}
              </button>
            </div>
          </div>
        )}

        {/* Display Captured Images */}
        {(capturedImages.front || capturedImages.back || uploadedImages.front || uploadedImages.back) && (
          <div className="mb-8 flex flex-col sm:flex-row sm:space-x-6 space-y-6 sm:space-y-0 justify-center">
            {(capturedImages.front || uploadedImages.front) && (
              <div className="text-center">
                <h2 className="text-lg font-semibold text-gray-800 mb-2">Front Image</h2>
                <img
                  src={capturedImages.front || uploadedImages.front}
                  alt="Front card"
                  className="card-image mx-auto rounded-lg shadow-md w-full max-w-xs"
                />
              </div>
            )}
            {(capturedImages.back || uploadedImages.back) && (
              <div className="text-center">
                <h2 className="text-lg font-semibold text-gray-800 mb-2">Back Image</h2>
                <img
                  src={capturedImages.back || uploadedImages.back}
                  alt="Back card"
                  className="card-image mx-auto rounded-lg shadow-md w-full max-w-xs"
                />
              </div>
            )}
          </div>
        )}

        {/* Display Extracted Details */}
        {(capturedImages.front || uploadedImages.front || capturedImages.back || uploadedImages.back) && currentStep !== 'upload' && currentStep !== 'proceed' && (
          <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-indigo-500">
            {/* Processing Time */}
            <div className="mb-6">
              <p className="text-gray-700 flex items-center">
                <strong className="mr-2 text-lg font-medium">Total Processing Time:</strong> {totalProcessingTime.toFixed(2)} seconds
                {totalProcessingTime > 3 && (
                  <span className="text-red-600 ml-2 flex items-center text-sm">
                    <i className="fas fa-exclamation-circle mr-1"></i>(Exceeds 3s target)
                  </span>
                )}
              </p>
            </div>

            {/* Front Details */}
            {(capturedImages.front || uploadedImages.front) && (
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4 text-center flex items-center justify-center">
                  <FaCreditCard className="text-indigo-600 mr-2" />
                  Card Details (Front)
                </h2>
                <div className="space-y-4">
                  <div className="flex items-center bg-gray-50 p-3 rounded-lg shadow-inner">
                    <FaCreditCard className="text-indigo-600 mr-3 text-xl" />
                    <p className="text-gray-700 flex-1">
                      <strong>Card Number:</strong> {frontDetails.final.card_number}
                      <span className="text-sm text-gray-500 ml-2">
                        ({frontDetails.used_vit.card_number ? 'ViT' : 'Vision API'}, Confidence: {frontDetails.confidence_scores.card_number.toFixed(2)})
                      </span>
                    </p>
                  </div>
                  <div className="flex items-center bg-gray-50 p-3 rounded-lg shadow-inner">
                    <FaCalendarAlt className="text-indigo-600 mr-3 text-xl" />
                    <p className="text-gray-700 flex-1">
                      <strong>Expiry Date:</strong> {frontDetails.final.expiry_date}
                      <span className="text-sm text-gray-500 ml-2">
                        ({frontDetails.used_vit.expiry_date ? 'ViT' : 'Vision API'}, Confidence: {frontDetails.confidence_scores.expiry_date.toFixed(2)})
                      </span>
                    </p>
                  </div>
                  <div className="flex items-center bg-gray-50 p-3 rounded-lg shadow-inner">
                    <p className="text-gray-700 flex-1">
                      <strong>Card Type:</strong> {frontDetails.final.card_type}
                    </p>
                  </div>
                  {frontDetails.final.errors && frontDetails.final.errors.length > 0 && (
                    <div className="text-red-600 bg-red-50 p-3 rounded-lg">
                      <strong>Errors:</strong>
                      <ul className="list-disc pl-5">
                        {frontDetails.final.errors.map((error, index) => (
                          <li key={index}>{error}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                {currentStep === 'results' && (
                  <div className="flex justify-center mt-4">
                    <button
                      onClick={() => {
                        setShowFeedback(true);
                        setFeedbackSide('front');
                      }}
                      className="btn-primary w-full sm:w-auto px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-800 text-white rounded-lg shadow-md hover:from-indigo-700 hover:to-indigo-900 transition-all duration-200"
                    >
                      Provide Feedback (Front)
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Back Details */}
            {(capturedImages.back || uploadedImages.back) && (
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-800 mb-4 text-center flex items-center justify-center">
                  <FaLock className="text-indigo-600 mr-2" />
                  Card Details (Back)
                </h2>
                <div className="space-y-4">
                  <div className="flex items-center bg-gray-50 p-3 rounded-lg shadow-inner">
                    <FaLock className="text-indigo-600 mr-3 text-xl" />
                    <p className="text-gray-700 flex-1">
                      <strong>Security Number:</strong> {backDetails.final.security_number}
                      <span className="text-sm text-gray-500 ml-2">
                        ({backDetails.used_vit.security_number ? 'ViT' : 'Vision API'}, Confidence: {backDetails.confidence_scores.security_number.toFixed(2)})
                      </span>
                    </p>
                  </div>
                  {backDetails.final.errors && backDetails.final.errors.length > 0 && (
                    <div className="text-red-600 bg-red-50 p-3 rounded-lg">
                      <strong>Errors:</strong>
                      <ul className="list-disc pl-5">
                        {backDetails.final.errors.map((error, index) => (
                          <li key={index}>{error}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                {currentStep === 'results' && (
                  <div className="flex justify-center mt-4">
                    <button
                      onClick={() => {
                        setShowFeedback(true);
                        setFeedbackSide('back');
                      }}
                      className="btn-primary w-full sm:w-auto px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-800 text-white rounded-lg shadow-md hover:from-indigo-700 hover:to-indigo-900 transition-all duration-200"
                    >
                      Provide Feedback (Back)
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Feedback Form */}
            {showFeedback && (
              <div className="mt-6 bg-gray-50 p-6 rounded-xl shadow-lg">
                <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center">
                  Provide Feedback ({feedbackSide})
                </h2>
                {feedbackSide === 'front' ? (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-gray-700 font-medium mb-1">Card Number</label>
                      <input
                        type="text"
                        value={feedback.card_number}
                        onChange={(e) => setFeedback({ ...feedback, card_number: e.target.value })}
                        className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all bg-white shadow-sm"
                        placeholder="Enter corrected card number"
                      />
                    </div>
                    <div>
                      <label className="block text-gray-700 font-medium mb-1">Expiry Date (MM/YY)</label>
                      <input
                        type="text"
                        value={feedback.expiry_date}
                        onChange={(e) => setFeedback({ ...feedback, expiry_date: e.target.value })}
                        className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all bg-white shadow-sm"
                        placeholder="Enter corrected expiry date"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <label className="block text-gray-700 font-medium mb-1">Security Number</label>
                      <input
                        type="text"
                        value={feedback.security_number}
                        onChange={(e) => setFeedback({ ...feedback, security_number: e.target.value })}
                        className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all bg-white shadow-sm"
                        placeholder="Enter corrected security number"
                      />
                    </div>
                  </div>
                )}
                <div className="mt-6 flex space-x-4 justify-center">
                  <button
                    onClick={submitFeedback}
                    className="btn-secondary w-full sm:w-auto px-6 py-3 bg-white text-indigo-600 border-2 border-indigo-600 rounded-lg shadow-md hover:bg-indigo-50 transition-all duration-200"
                  >
                    Submit Feedback
                  </button>
                  <button
                    onClick={() => setShowFeedback(false)}
                    className="btn-cancel w-full sm:w-auto px-6 py-3 bg-red-600 text-white rounded-lg shadow-md hover:bg-red-700 transition-all duration-200"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* Confirmation Form */}
            {showConfirmation && confirmationData && (
              <div className="mt-6 bg-gray-50 p-6 rounded-xl shadow-lg">
                <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center">
                  Confirm Feedback ({confirmationData.side})
                </h2>
                <div className="space-y-2 text-gray-800">
                  {confirmationData.side === 'front' ? (
                    <>
                      <p className="bg-gray-50 p-3 rounded-lg shadow-inner">
                        <strong>Card Number:</strong> {confirmationData.corrected_result.card_number}
                      </p>
                      <p className="bg-gray-50 p-3 rounded-lg shadow-inner">
                        <strong>Expiry Date:</strong> {confirmationData.corrected_result.expiry_date}
                      </p>
                    </>
                  ) : (
                    <p className="bg-gray-50 p-3 rounded-lg shadow-inner">
                      <strong>Security Number:</strong> {confirmationData.corrected_result.security_number}
                    </p>
                  )}
                  {confirmationData.corrected_result.errors && confirmationData.corrected_result.errors.length > 0 && (
                    <div className="text-red-600 bg-red-50 p-3 rounded-lg">
                      <strong>Errors:</strong>
                      <ul className="list-disc pl-5">
                        {confirmationData.corrected_result.errors.map((error: string, index: number) => (
                          <li key={index}>{error}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
                <div className="mt-6 flex space-x-4 justify-center">
                  <button
                    onClick={confirmFeedback}
                    className="btn-secondary w-full sm:w-auto px-6 py-3 bg-white text-indigo-600 border-2 border-indigo-600 rounded-lg shadow-md hover:bg-indigo-50 transition-all duration-200"
                  >
                    Confirm
                  </button>
                  <button
                    onClick={() => {
                      setShowConfirmation(false);
                      setCurrentStep('results');
                    }}
                    className="btn-cancel w-full sm:w-auto px-6 py-3 bg-red-600 text-white rounded-lg shadow-md hover:bg-red-700 transition-all duration-200"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* Proceed Button */}
            {(currentStep === 'results' || currentStep === 'confirmation') && (
              <div className="mt-8 flex justify-center">
                <button
                  onClick={handleProceed}
                  className="btn-secondary w-full sm:w-auto px-6 py-3 bg-white text-indigo-600 border-2 border-indigo-600 rounded-lg shadow-md hover:bg-indigo-50 transition-all duration-200"
                >
                  Proceed
                </button>
              </div>
            )}
          </div>
        )}

        {/* Proceed Confirmation */}
        {currentStep === 'proceed' && (
          <div className="bg-white p-6 rounded-xl shadow-lg border-t-4 border-green-500 text-center">
            <h2 className="text-2xl font-semibold text-green-800 mb-4 flex items-center justify-center">
              <FaCheckCircle className="text-green-600 mr-2" />
              Extraction Confirmed
            </h2>
            <p className="text-gray-700 mb-6 text-lg">
              All data has been processed and cleared. You can start a new scan.
            </p>
            <button
              onClick={() => {
                setCurrentStep('upload');
                setShowWebcam(false);
                setIsCapturingFront(true);
              }}
              className="btn-primary w-full sm:w-auto px-6 py-3 bg-gradient-to-r from-green-600 to-green-800 text-white rounded-lg shadow-md hover:from-green-700 hover:to-green-900 transition-all duration-200"
            >
              Start New Scan
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;