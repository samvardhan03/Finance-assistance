import React, { useState } from 'react';
import UserInput from './UserInput';
import AdviceOutput from './AdviceOutput';
import { generateAdvice } from '../services/api';

const FinancialAssistant = () => {
  const [userInput, setUserInput] = useState({});
  const [advice, setAdvice] = useState('');

  const handleSubmit = async (input) => {
    setUserInput(input);
    const advice = await generateAdvice(input);
    setAdvice(advice);
  };

  return (
    <div>
      <UserInput onSubmit={handleSubmit} />
      <AdviceOutput advice={advice} />
    </div>
  );
};

export default FinancialAssistant;