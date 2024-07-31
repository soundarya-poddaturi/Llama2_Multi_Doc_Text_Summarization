// App.js
import { Routes, Route } from 'react-router-dom';
import './App.css';
import Main from './components/Main';

const App = () => {
 return (
    <>
       <Routes>
          <Route path="/" element={<Main/>} />
       </Routes>
    </>
 );
};

export default App;