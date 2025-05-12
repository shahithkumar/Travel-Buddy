const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors());

const db = new sqlite3.Database('./users.db');
const SECRET_KEY = 'mysecretkey123'; // Must match Flask

// Create users table
db.run(`CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    gender TEXT,
    password TEXT NOT NULL
)`);

app.post('/register', (req, res) => {
    const { name, email, phone, gender, password } = req.body;
    const hashedPassword = bcrypt.hashSync(password, 8);

    db.run(
        `INSERT INTO users (name, email, phone, gender, password) VALUES (?, ?, ?, ?, ?)`,
        [name, email, phone, gender, hashedPassword],
        (err) => {
            if (err) return res.status(500).send('Error registering user: ' + err.message);
            res.status(200).send('User registered successfully');
        }
    );
});

app.post('/login', (req, res) => {
    const { email, password } = req.body;

    db.get(`SELECT * FROM users WHERE email = ?`, [email], (err, user) => {
        if (err || !user) return res.status(404).send('User not found');
        const passwordIsValid = bcrypt.compareSync(password, user.password);
        if (!passwordIsValid) return res.status(401).send('Invalid password');

        const token = jwt.sign({ id: user.id }, SECRET_KEY, { expiresIn: '1h' });
        res.status(200).json({ token });
    });
});

app.listen(3000, () => console.log('Auth server running on port 3000'));