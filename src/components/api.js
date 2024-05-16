import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

export const generateAdvice = async (userData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/generate_advice`, userData);
    return response.data.advice;
  } catch (error) {
    console.error('Error generating advice:', error);
    return 'Error generating advice';
  }
};