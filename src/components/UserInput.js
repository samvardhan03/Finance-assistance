import React, { useState } from 'react';

const UserInput = ({ onSubmit }) => {
  const [formData, setFormData] = useState({});

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Add form fields for user input */}
      <input
        type="text"
        name="customerID"
        placeholder="Customer ID"
        onChange={handleChange}
      />
      {/* Add more input fields as needed */}
      <button type="submit">Generate Advice</button>
    </form>
  );
};

export default UserInput;