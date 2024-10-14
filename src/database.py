from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker



Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # Define the ID column
    text = Column(Text, nullable=True)  # Use Text for the 'text' column

# Database URL configuration
DATABASE_URL = "mysql+pymysql://root:9900@localhost/plagiarism_db"

# Create an engine instance
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



# Function to add a document to the database
def add_to_database(text):
    session = SessionLocal()  # Create a new session
    try:
        new_doc = Document(text=text)
        session.add(new_doc)
        session.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()  # Always close the session to release resources
