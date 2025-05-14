import React, { useState } from 'react';
import axios from 'axios';
import './CardScanner.css';

const CardScanner = () => {
  const [imageFront, setImageFront] = useState(null);
  const [imageBack, setImageBack] = useState(null);
  const [frontResult, setFrontResult] = useState(null);
  const [backResult, setBackResult] = useState(null);
  const [processingTimeFront, setProcessingTimeFront] = useState(null);
  const [processingTimeBack, setProcessingTimeBack] = useState(null);
  const [errorFront, setErrorFront] = useState(null);
  const [errorBack, setErrorBack] = useState(null);
  const [correctionsFront, setCorrectionsFront] = useState({});
  const [correctionsBack, setCorrectionsBack] = useState({});
  const [showFeedbackFront, setShowFeedbackFront] = useState(false);
  const [showFeedbackBack, setShowFeedbackBack] = useState(false);
  const [feedbackMessageFront, setFeedbackMessageFront] = useState(null);
  const [feedbackMessageBack, setFeedbackMessageBack] = useState(null);
  const [imagePathFront, setImagePathFront] = useState(null);
  const [imagePathBack, setImagePathBack] = useState(null);
  const [proceedMessage, setProceedMessage] = useState(null);

  const handleImageUpload = (event, side) => {
    const file = event.target.files[0];
    if (file) {
      if (side === 'front') {
        setImageFront(URL.createObjectURL(file));
        processImage(file, 'front');
      } else {
        setImageBack(URL.createObjectURL(file));
        processImage(file, 'back');
      }
    }
  };

  const processImage = async (file, side) => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('side', side);

    try {
      const response = await axios.post('http://localhost:5000/ocr', formData);
      console.log(`${side} response:`, response.data);

      const { final_result, processing_time, vision_result, errors } = response.data;

      if (side === 'front') {
        setFrontResult(final_result);
        setProcessingTimeFront(processing_time);
        setErrorFront(errors && errors.length > 0 ? errors.join(', ') : null);
        setShowFeedbackFront(true);
        setImagePathFront('temp_card.jpg');
      } else {
        setBackResult(final_result);
        setProcessingTimeBack(processing_time);
        setErrorBack(errors && errors.length > 0 ? errors.join(', ') : null);
        setShowFeedbackBack(true);
        setImagePathBack('temp_card.jpg');
      }
    } catch (error) {
      console.error(`Error processing ${side} image:`, error);
      if (side === 'front') {
        setErrorFront('Failed to process image: Failed to fetch');
        setFrontResult(null);
        setProcessingTimeFront(null);
      } else {
        setErrorBack('Failed to process image: Failed to fetch');
        setBackResult(null);
        setProcessingTimeBack(null);
      }
    }
  };

  const handleFeedbackSubmit = async (side) => {
    const corrections = side === 'front' ? correctionsFront : correctionsBack;
    const imagePath = side === 'front' ? imagePathFront : imagePathBack;

    try {
      const response = await axios.post('http://localhost:5000/feedback', {
        image_path: imagePath,
        corrections,
        side
      });
      console.log(`${side} feedback response:`, response.data);

      if (side === 'front') {
        setFeedbackMessageFront(response.data.message);
      } else {
        setFeedbackMessageBack(response.data.message);
      }
    } catch (error) {
      console.error(`Error submitting ${side} feedback:`, error);
      if (side === 'front') {
        setFeedbackMessageFront('Failed to submit feedback');
      } else {
        setFeedbackMessageBack('Failed to submit feedback');
      }
    }
  };

  const handleFeedbackConfirm = async (side) => {
    const corrections = side === 'front' ? correctionsFront : correctionsBack;
    const imagePath = side === 'front' ? imagePathFront : imagePathBack;

    try {
      const response = await axios.post('http://localhost:5000/confirm_feedback', {
        image_path: imagePath,
        corrections,
        side
      });
      console.log(`${side} confirm feedback response:`, response.data);

      if (side === 'front') {
        setFeedbackMessageFront(response.data.message);
      } else {
        setFeedbackMessageBack(response.data.message);
      }
    } catch (error) {
      console.error(`Error confirming ${side} feedback:`, error);
      if (side === 'front') {
        setFeedbackMessageFront('Failed to confirm feedback');
      } else {
        setFeedbackMessageBack('Failed to confirm feedback');
      }
    }
  };

  const handleProceed = async () => {
    try {
      const response = await axios.post('http://localhost:5000/proceed');
      console.log('Proceed response:', response.data);
      setProceedMessage(response.data.message);
      setImageFront(null);
      setImageBack(null);
      setFrontResult(null);
      setBackResult(null);
      setProcessingTimeFront(null);
      setProcessingTimeBack(null);
      setErrorFront(null);
      setErrorBack(null);
      setCorrectionsFront({});
      setCorrectionsBack({});
      setShowFeedbackFront(false);
      setShowFeedbackBack(false);
      setFeedbackMessageFront(null);
      setFeedbackMessageBack(null);
      setImagePathFront(null);
      setImagePathBack(null);
    } catch (error) {
      console.error('Error proceeding:', error);
      setProceedMessage('Failed to proceed');
    }
  };

  return (
    <div className="card-scanner">
      <h1>Credit Card Scanner</h1>

      <div className="side">
        <h2>Front Side</h2>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => handleImageUpload(e, 'front')}
        />
        {imageFront && <img src={imageFront} alt="Front Preview" className="preview" />}
        {processingTimeFront && (
          <p>Processing Time: {processingTimeFront.toFixed(2)} seconds {processingTimeFront > 3 ? '(Exceeds 3s target)' : ''}</p>
        )}
        {errorFront && <p className="error">Errors: {errorFront}</p>}
        {frontResult && (
          <div>
            <h3>Card Details (Front)</h3>
            <p>Card Number: {frontResult.card_number || 'N/A'} (Vision API, Confidence: {frontResult.card_number ? 0.95 : 0.00})</p>
            <p>Expiry Date: {frontResult.expiry_date || 'N/A'} (Vision API, Confidence: {frontResult.expiry_date ? 0.95 : 0.00})</p>
            <p>Card Type: {frontResult.card_type || 'Unknown'}</p>
            {showFeedbackFront && (
              <div>
                <h4>Provide Feedback (Front)</h4>
                <input
                  type="text"
                  placeholder="Correct Card Number"
                  onChange={(e) => setCorrectionsFront({ ...correctionsFront, card_number: e.target.value })}
                />
                <input
                  type="text"
                  placeholder="Correct Expiry Date (MM/YY)"
                  onChange={(e) => setCorrectionsFront({ ...correctionsFront, expiry_date: e.target.value })}
                />
                <button onClick={() => handleFeedbackSubmit('front')}>Submit Feedback</button>
                {feedbackMessageFront && (
                  <div>
                    <p>{feedbackMessageFront}</p>
                    {feedbackMessageFront.includes('Please confirm') && (
                      <button onClick={() => handleFeedbackConfirm('front')}>Confirm Feedback</button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="side">
        <h2>Back Side</h2>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => handleImageUpload(e, 'back')}
        />
        {imageBack && <img src={imageBack} alt="Back Preview" className="preview" />}
        {processingTimeBack && (
          <p>Processing Time: {processingTimeBack.toFixed(2)} seconds {processingTimeBack > 3 ? '(Exceeds 3s target)' : ''}</p>
        )}
        {errorBack && <p className="error">Errors: {errorBack}</p>}
        {backResult && (
          <div>
            <h3>Card Details (Back)</h3>
            <p>Security Number: {backResult.security_number || 'N/A'} (Vision API, Confidence: {backResult.security_number ? 0.95 : 0.00})</p>
            {showFeedbackBack && (
              <div>
                <h4>Provide Feedback (Back)</h4>
                <input
                  type="text"
                  placeholder="Correct Security Number"
                  onChange={(e) => setCorrectionsBack({ ...correctionsBack, security_number: e.target.value })}
                />
                <button onClick={() => handleFeedbackSubmit('back')}>Submit Feedback</button>
                {feedbackMessageBack && (
                  <div>
                    <p>{feedbackMessageBack}</p>
                    {feedbackMessageBack.includes('Please confirm') && (
                      <button onClick={() => handleFeedbackConfirm('back')}>Confirm Feedback</button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {(feedbackMessageFront || feedbackMessageBack) && (
        <div>
          <button onClick={handleProceed}>Proceed</button>
          {proceedMessage && <p>{proceedMessage}</p>}
        </div>
      )}
    </div>
  );
};

export default CardScanner;