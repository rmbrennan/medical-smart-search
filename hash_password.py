import bcrypt
import sys

def generate_hash(password):
    # Generate salt and hash with 12 rounds (default for streamlit-authenticator)
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hash_password.py <password>")
        sys.exit(1)

    password = sys.argv[1]
    hashed = generate_hash(password)
    print(f"Hashed password: {hashed}")