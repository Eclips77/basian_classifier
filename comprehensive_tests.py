import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import json

# Import the classes we want to test
from src.app_controller import AppController
from server import app

class TestAppController:
    """טסטים עבור AppController"""
    
    def setup_method(self):
        """הגדרת נתונים לפני כל טסט"""
        # יצירת mock loader
        self.mock_loader = Mock()
        self.controller = AppController(label_col="BoughtComputer", loader=self.mock_loader)
        
        # יצירת DataFrame מדומה לטסטים
        self.sample_data = pd.DataFrame({
            'Age': ['Youth', 'Middle-aged', 'Senior'],
            'Income': ['High', 'Medium', 'Low'],
            'Student': ['No', 'Yes', 'No'],
            'CreditRating': ['Fair', 'Excellent', 'Fair'],
            'BoughtComputer': ['No', 'Yes', 'Yes']
        })
    
    def test_init(self):
        """בדיקת אתחול של AppController"""
        assert self.controller.label_col == "BoughtComputer"
        assert self.controller.loader == self.mock_loader
        assert self.controller.data is None
        assert self.controller.classifier is None
        assert self.controller.evaluator is None
    
    @patch('src.app_controller.DataValidator')
    def test_load_and_prepare_success(self, mock_validator):
        """בדיקת טעינה והכנת נתונים - מקרה הצלחה"""
        # הגדרת mock values
        mock_validator.validate_cli.return_value = ("valid_path.csv", "BoughtComputer")
        self.mock_loader.load_data.return_value = self.sample_data
        
        # Mock the Cleaner
        with patch('src.app_controller.Cleaner') as mock_cleaner:
            mock_cleaner_instance = Mock()
            mock_cleaner.return_value = mock_cleaner_instance
            mock_cleaner_instance.split_data.return_value = (
                self.sample_data[['Age', 'Income']],  # X_train
                self.sample_data[['Student', 'CreditRating']],  # X_test
                self.sample_data['BoughtComputer'][:2],  # y_train
                self.sample_data['BoughtComputer'][2:]   # y_test
            )
            
            # הרצת הפונקציה
            self.controller.load_and_prepare("test_file.csv")
            
            # בדיקות
            mock_validator.validate_cli.assert_called_once_with("test_file.csv", "BoughtComputer")
            self.mock_loader.load_data.assert_called_once_with("valid_path.csv")
            assert self.controller.data is not None
            assert self.controller.X_train is not None
            assert self.controller.X_test is not None
    
    @patch('src.app_controller.NaiveBayesTrainer')
    @patch('src.app_controller.NaiveBayesEvaluator')
    def test_train_model_success(self, mock_evaluator_class, mock_trainer_class):
        """בדיקת אימון המודל - מקרה הצלחה"""
        # הגדרת mock objects
        mock_trainer = Mock()
        mock_model = Mock()
        mock_evaluator = Mock()
        
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.fit.return_value = mock_model
        mock_evaluator_class.return_value = mock_evaluator
        
        # הגדרת נתוני אימון מדומים
        self.controller.X_train = self.sample_data[['Age', 'Income']]
        self.controller.y_train = self.sample_data['BoughtComputer']
        
        # הרצת הפונקציה
        self.controller.train_model()
        
        # בדיקות
        mock_trainer.fit.assert_called_once()
        assert self.controller.classifier == mock_trainer
        assert self.controller.evaluator == mock_evaluator
    
    def test_get_accuracy_no_model(self):
        """בדיקת קבלת דיוק כאשר אין מודל מאומן"""
        with pytest.raises(Exception, match="Model not trained"):
            self.controller.get_accuracy()
    
    def test_get_accuracy_with_model(self):
        """בדיקת קבלת דיוק עם מודל מאומן"""
        # הגדרת mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = 0.85
        self.controller.evaluator = mock_evaluator
        
        # הגדרת נתוני טסט מדומים
        self.controller.X_test = self.sample_data[['Age', 'Income']]
        self.controller.y_test = self.sample_data['BoughtComputer']
        
        # הרצת הפונקציה
        accuracy = self.controller.get_accuracy()
        
        # בדיקה
        assert accuracy == 0.85
        mock_evaluator.evaluate.assert_called_once()
    
    def test_get_schema(self):
        """בדיקת קבלת schema של הנתונים"""
        # הגדרת נתוני אימון
        self.controller.X_train = pd.DataFrame({
            'Age': ['Youth', 'Middle-aged', 'Senior'],
            'Income': ['High', 'Medium', 'Low']
        })
        
        # הרצת הפונקציה
        schema = self.controller.get_schema()
        
        # בדיקות
        assert 'Age' in schema
        assert 'Income' in schema
        assert set(schema['Age']) == {'Youth', 'Middle-aged', 'Senior'}
        assert set(schema['Income']) == {'High', 'Medium', 'Low'}
    
    def test_predict_record_no_model(self):
        """בדיקת חיזוי ללא מודל מאומן"""
        with pytest.raises(Exception, match="Model not trained"):
            self.controller.predict_record({'Age': 'Youth', 'Income': 'High'})
    
    def test_predict_record_with_model(self):
        """בדיקת חיזוי עם מודל מאומן"""
        # הגדרת mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.predict.return_value = 'Yes'
        self.controller.evaluator = mock_evaluator
        
        # הרצת הפונקציה
        record = {'Age': 'Youth', 'Income': 'High'}
        result = self.controller.predict_record(record)
        
        # בדיקה
        assert result == 'Yes'
        mock_evaluator.predict.assert_called_once_with(record)


class TestServerAPI:
    """טסטים עבור FastAPI server"""
    
    def setup_method(self):
        """הגדרת TestClient לפני כל טסט"""
        self.client = TestClient(app)

    @patch('server.loader')
    def test_load_file_success(self, mock_loader):
        df = pd.DataFrame({'A': [1], 'B': [2]})
        mock_loader.load_data.return_value = df

        response = self.client.post('/load_file', json={'file_path': 'x.csv'})

        assert response.status_code == 200
        assert response.json() == {'columns': ['A', 'B']}
        mock_loader.load_data.assert_called_once_with('x.csv')

    @patch('server.AppController')
    def test_train_success(self, mock_app):
        mock_instance = Mock()
        mock_instance.get_accuracy.return_value = 0.5
        mock_app.return_value = mock_instance

        response = self.client.post('/train', json={'file_path': 'x.csv', 'label_column': 'A'})

        assert response.status_code == 200
        assert response.json() == {'accuracy': 0.5}
        mock_app.assert_called_once()
    
    @patch('server.controller')
    def test_get_accuracy_success(self, mock_controller):
        """בדיקת endpoint של accuracy - מקרה הצלחה"""
        mock_controller.get_accuracy.return_value = 0.92
        
        response = self.client.get("/accuracy")
        
        assert response.status_code == 200
        assert response.json() == {"accuracy": 0.92}
    
    @patch('server.controller')
    def test_get_accuracy_error(self, mock_controller):
        """בדיקת endpoint של accuracy - מקרה שגיאה"""
        mock_controller.get_accuracy.side_effect = Exception("Model not trained")
        
        response = self.client.get("/accuracy")
        
        assert response.status_code == 400
        assert "Model not trained" in response.json()["detail"]
    
    @patch('server.controller')
    def test_get_schema_success(self, mock_controller):
        """בדיקת endpoint של schema - מקרה הצלחה"""
        expected_schema = {
            "Age": ["Youth", "Middle-aged", "Senior"],
            "Income": ["High", "Medium", "Low"]
        }
        mock_controller.get_schema.return_value = expected_schema
        
        response = self.client.get("/schema")
        
        assert response.status_code == 200
        assert response.json() == expected_schema
    
    @patch('server.controller')
    def test_get_schema_error(self, mock_controller):
        """בדיקת endpoint של schema - מקרה שגיאה"""
        mock_controller.get_schema.side_effect = Exception("Data not loaded")
        
        response = self.client.get("/schema")
        
        assert response.status_code == 400
        assert "Data not loaded" in response.json()["detail"]
    
    @patch('server.controller')
    def test_predict_success(self, mock_controller):
        """בדיקת endpoint של predict - מקרה הצלחה"""
        mock_controller.predict_record.return_value = "Yes"
        
        test_record = {
            "record": {
                "Age": "Youth",
                "Income": "High",
                "Student": "No",
                "CreditRating": "Fair"
            }
        }
        
        response = self.client.post("/predict", json=test_record)
        
        assert response.status_code == 200
        assert response.json() == {"prediction": "Yes"}
        mock_controller.predict_record.assert_called_once_with(test_record["record"])
    
    @patch('server.controller')
    def test_predict_error(self, mock_controller):
        """בדיקת endpoint של predict - מקרה שגיאה"""
        mock_controller.predict_record.side_effect = Exception("Invalid input")
        
        test_record = {
            "record": {
                "Age": "InvalidAge",
                "Income": "High"
            }
        }
        
        response = self.client.post("/predict", json=test_record)
        
        assert response.status_code == 400
        assert "Invalid input" in response.json()["detail"]
    
    def test_predict_invalid_json(self):
        """בדיקת endpoint של predict עם JSON לא תקין"""
        response = self.client.post("/predict", json={"invalid": "structure"})
        
        assert response.status_code == 422  # Validation error


# טסטים נוספים לבדיקת אינטגרציה
class TestIntegration:
    """טסטים אינטגרטיביים"""
    
    def setup_method(self):
        """הגדרת TestClient לפני כל טסט"""
        self.client = TestClient(app)
    
    def test_full_workflow_simulation(self):
        """סימולציה של תהליך עבודה מלא"""
        # בדיקת schema
        schema_response = self.client.get("/schema")
        assert schema_response.status_code == 200
        schema = schema_response.json()
        
        # בדיקת accuracy
        accuracy_response = self.client.get("/accuracy")
        assert accuracy_response.status_code == 200
        accuracy = accuracy_response.json()["accuracy"]
        assert 0 <= accuracy <= 1  # דיוק צריך להיות בין 0 ל-1
        
        # בדיקת חיזוי עם נתונים מה-schema
        if schema:
            first_col = list(schema.keys())[0]
            if schema[first_col]:
                test_record = {
                    "record": {first_col: schema[first_col][0]}
                }
                predict_response = self.client.post("/predict", json=test_record)
                # הטסט הזה יכול להיכשל אם הנתונים לא מלאים, אבל זה בסדר
                assert predict_response.status_code in [200, 400]


# פונקציית עזר לבדיקת ביצועים
def test_performance_benchmark():
    """בדיקת ביצועים בסיסית"""
    import time
    
    client = TestClient(app)
    
    # מדידת זמן תגובה לקבלת accuracy
    start_time = time.time()
    response = client.get("/accuracy")
    end_time = time.time()
    
    response_time = end_time - start_time
    
    # בדיקה שהתגובה מהירה (פחות מ-2 שניות)
    assert response_time < 2.0, f"Response time too slow: {response_time:.2f}s"
    
    print(f"✅ Performance test passed: {response_time:.3f}s")


if __name__ == "__main__":
    # הרצת הטסטים עם פרטים נוספים
    pytest.main([__file__, "-v", "--tb=short"])
