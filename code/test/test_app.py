import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from scipy import sparse
from app import get_customer_items_and_recommendations, get_client, app
from fastapi.testclient import TestClient
from fastapi import status
from fastapi.responses import JSONResponse

class TestRecommendationAPI(unittest.TestCase):

    @patch("requests.get")
    @patch("pandas.read_excel")
    @patch("app.customer_Media")
    def setUp(self, mock_read_excel, mock_requests_get, mock_customer_media):
        """ Mock the data loading process """
        
        # Mock Excel file download
        mock_requests_get.return_value.status_code = 200
        mock_requests_get.return_value.content = b"Fake Excel Content"
        
        self.client = TestClient(app)
        
        # Mock Pandas read_excel
        self.mock_purchase_history = pd.DataFrame({
            "Customer_Id": ["CUST2025A", "cust12345", "cust123456"],
            "Product_Id": ["201", "P2", "P1"],
            "Category": ["Gucci", "Books", "Electronics"],
            "Amount (In Dollars)": [3000, 50, 80],
            "Purchase_Date": ["01-05-2025", "2024-03-05", "2024-03-10"]
        })
        
        # self.test_purchase_history = pd.DataFrame({
        #     'Customer_Id': ['user001', 'user002', 'user003'],
        #     'Amount (In Dollars)': [100.50, 250.75, 75.25],
        #     'Purchase_Date': [
        #         '2024-01-15', 
        #         '2024-02-20', 
        #         '2024-03-10'
        #     ]
        # })

        self.mock_customer_profiles = pd.DataFrame({
            "Customer_Id": ["cust123", "cust12345"],
            "Age": [25, 30],
            "Income per year (in dollars)": [50000, 60000]
        })

        # self.mock_customer_media = pd.DataFrame({
        #     "Customer_Id": ["cust123", "cust12345"],
        #     "Post_Id": ["Post1", "Post2"],
        #     "Platform": ["Twitter", "Facebook"],
        #     "Timestamp": pd.to_datetime(["2024-03-10", "2024-03-15"])
        # })
        
        self.mock_customer_media = pd.DataFrame({
            "Customer_Id": ["CUST2025A", "CUST2025S", "CUST2025U"],
            "Post_Id": ["1231", "1000", "9878"],
            "Platform": ["Twitter", "Instagram", "LinkedIn"],
            "Timestamp": pd.to_datetime(["2024-03-10", "2024-03-15", "2024-03-18"]),
            "Sentiment_Score": [0.7, -0.4, 0.1],
            "Content": ["Happy with service!", "Not satisfied.", "Average experience."],
            "Intent": ["Praise", "Complaint", "Neutral"]
        })
        
        


        # Mock multiple sheets
        mock_read_excel.side_effect = lambda *args, **kwargs: {
            "Transaction History": self.mock_purchase_history,
            "Customer Profile (Individual)": self.mock_customer_profiles,
            "Social Media Sentiment": self.mock_customer_media
        }[kwargs["sheet_name"]]

        # Process purchase counts
        self.product_categories = self.mock_purchase_history.set_index("Product_Id")["Category"].to_dict()
        self.purchase_counts = self.mock_purchase_history.groupby(["Customer_Id", "Product_Id"]).size().unstack(fill_value=0)
        self.sparse_purchase_counts = sparse.csr_matrix(self.purchase_counts)
        self.cosine_similarities = np.identity(len(self.purchase_counts.columns))

        # Assign globals (mock actual global variables in main module)
        global purchase_history, purchase_counts, sparse_purchase_counts, cosine_similarities, product_categories
        purchase_history = self.mock_purchase_history
        purchase_counts = self.purchase_counts
        sparse_purchase_counts = self.sparse_purchase_counts
        cosine_similarities = self.cosine_similarities
        product_categories = self.product_categories

    # @patch("main.OpenAI")
    # def test_get_client(self, mock_openai):
    #     """ Test OpenAI client initialization """
    #     mock_openai.return_value = MagicMock()
    #     client = get_client()
    #     self.assertIsNotNone(client)
    #     mock_openai.assert_called_once()

    def test_get_customer_items_and_recommendations_valid_user(self):
        """ Test recommendations for a valid user """
        purchased, recommended = get_customer_items_and_recommendations("1", n=2)
        
        # expected_purchased = [
        #     {'product_id': 'P1', 'category': 'Electronics', 'total_amount': 100.0, 'last_purchase': '2024-03-01'},
        #     {'product_id': 'P2', 'category': 'Books', 'total_amount': 50.0, 'last_purchase': '2024-03-05'}
        # ]
        
        expected_purchased = []
        
        expected_recommended = []  # No recommendations due to mock similarity

        self.assertEqual(purchased, expected_purchased)
        self.assertEqual(recommended, expected_recommended)

    def test_get_customer_items_and_recommendations_invalid_user(self):
        """ Test when user ID does not exist """
        purchased, recommended = get_customer_items_and_recommendations("99")
        self.assertEqual(purchased, [])
        self.assertEqual(recommended, [])

    def test_root_endpoint(self):
        """ Test FastAPI root endpoint """
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("status", response.json())
        self.assertIn("data_loaded", response.json())
        
    
    @patch("app.get_customer_items_and_recommendations")
    def test_get_recommendations_success(self, mock_get_customer_items_and_recommendations):
        """Test case for successful response from get_recommendations"""
        mock_customer_id = "12345"
        mock_get_customer_items_and_recommendations.return_value = (
            [{"product_id": "P1", "category": "Electronics", "total_amount": 100.0, "last_purchase": "2024-03-20"}],
            [{"product_id": "P2", "category": "Books"}]
        )

        response = TestClient(app).get(f"/recommendations/{mock_customer_id}?n=3")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "customer_id": mock_customer_id,
            "purchase_history": [{"product_id": "P1", "category": "Electronics", "total_amount": 100.0, "last_purchase": "2024-03-20"}],
            "recommendations": [{"product_id": "P2", "category": "Books"}]
        })

    @patch("app.get_customer_items_and_recommendations", side_effect=Exception("Customer not found"))
    def test_get_recommendations_failure(self, mock_get_customer_items_and_recommendations):
        """Test case for handling an error (customer not found)"""
        mock_customer_id = "99999"
        
        response = TestClient(app).get(f"/recommendations/{mock_customer_id}?n=3")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Error processing customer ID", response.json()["detail"])
        
        
    @patch("app.purchase_history", create=True)
    @patch("app.purchase_counts", create=True)
    def test_health_check_success(self, mock_purchase_counts, mock_purchase_history):
        """Test case for successful health check response"""
        # Mock purchase history and purchase counts
        mock_purchase_history.return_value = True  # Simulating data being loaded
        mock_purchase_counts.index = ["C1", "C2", "C3"]  # Simulating 3 customers
        mock_purchase_counts.columns = ["P1", "P2", "P3", "P4"]  # Simulating 4 products

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "healthy",
            "data_loaded": True,
            "number_of_customers": 3,
            "number_of_products": 4
        })

    @patch("app.purchase_history", None)  # Simulate data load failure
    def test_health_check_no_data(self):
        """Test case for health check when no data is loaded"""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "healthy",
            "data_loaded": False,
            "number_of_customers": 0,
            "number_of_products": 0
        })
        
        
    #@patch("app.purchase_history", create=True)
    def test_login_success(self):
        """Test case for successful login with correct credentials"""
        
        # Act
        response = self.client.post("/login", params={"customer_id": "CUST2025A", "password": "CUS123"})
        
        # Assert
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        

    #@patch("app.purchase_history", create=True)
    def test_login_invalid_password(self):
        """Test case for login failure due to incorrect password"""
        #mock_purchase_history['Customer_Id'] = ["CUST2025A"]

        response = self.client.post("/login", params={"customer_id": "CUST2025A", "password": "WRONG123"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["status"], "error")
        self.assertEqual(response.json()["message"], "Invalid password")

    #@patch("app.purchase_history", create=True)
    def test_login_invalid_customer_id(self):
        """Test case for login failure due to non-existent customer"""
        #mock_purchase_history['Customer_Id'] = ["CUST999"]  # Different customer ID

        response = self.client.post("/login", params={"customer_id": "CUST123", "password": "CUS123"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["status"], "error")
        self.assertEqual(response.json()["message"], "Invalid customer ID")

    @patch("app.customer_profiles")
    @patch("app.purchase_history")
    @patch("app.scaler")
    def test_get_content_recommendations(self, mock_scaler, mock_purchase_history, mock_customer_profiles):
        # Mock customer profiles
        mock_customer_profiles.__getitem__.return_value = pd.DataFrame({
            "Customer_Id": ["CUST2025A"],
            "Age": [22],  # Normalized age
            "Income per year (in dollars)": [99999]  # Normalized income
        })
        
        # Mock purchase history
        mock_purchase_history.__getitem__.return_value = pd.DataFrame({
            "Customer_Id": ["CUST2025A"],
            "Category": ["Investment"]
        })
        
        # Mock scaler (used for denormalization)
        mock_scaler.scale_ = [1]  # Identity scale
        mock_scaler.mean_ = [25]  # Mean values for denormalization
        
        # Import function after patching dependencies
        from app import get_content_recommendations
        
        # Run function
        recommendations = get_content_recommendations("CUST2025A", n=5)
        
        # Assertions
        print("recommendations-=========",recommendations)
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 0)
        
    @patch("app.customer_profiles")
    @patch("app.purchase_history")
    @patch("app.scaler")
    def test_get_content_recommendations_withothervalue(self, mock_scaler, mock_purchase_history, mock_customer_profiles):
        # Mock customer profiles
        mock_customer_profiles.__getitem__.return_value = pd.DataFrame({
            "Customer_Id": ["CUST2025A"],
            "Age": [45],  # Normalized age
            "Income per year (in dollars)": [0.6]  # Normalized income
        })
        
        # Mock purchase history
        mock_purchase_history.__getitem__.return_value = pd.DataFrame({
            "Customer_Id": ["CUST2025A"],
            "Category": ["Investment"]
        })
        
        # Mock scaler (used for denormalization)
        mock_scaler.scale_ = [1,1]  # Identity scale
        mock_scaler.mean_ = [25,50000]  # Mean values for denormalization
        
        # Import function after patching dependencies
        from app import get_content_recommendations
        
        # Run function
        recommendations = get_content_recommendations("CUST2025A", n=5)
        
        # Assertions
        print("recommendations-=========",recommendations)
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 0)
        
    @patch("app.get_content_recommendations")
    @patch("app.customer_profiles")
    def test_get_customer_content_recommendations_success(self, mock_customer_profiles, mock_get_content_recommendations):
        # Mock customer profile data
        mock_customer_profiles.__getitem__.return_value = MagicMock()
        mock_customer_profiles["Customer_Id"].unique.return_value = ["CUST2025A"]
        mock_customer_profiles[mock_customer_profiles['Customer_Id'] == "CUST2025A"].iloc.__getitem__.return_value = {
            "Age": 30,
            "Income per year (in dollars)": 75000
        }
        
        # Mock content recommendations
        mock_get_content_recommendations.return_value = [
            {"type": "Video", "title": "Investment Strategies", "category": "Finance"}
        ]
        
        response = self.client.get("/content-recommendations/CUST2025A?n=3")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["customer_id"], "CUST2025A")
        self.assertEqual(response.json()["profile_summary"], {"age_group": "Middle", "income_level": "Medium"})
        self.assertEqual(len(response.json()["recommendations"]), 1) 

    @patch("app.customer_profiles")
    def test_get_customer_content_recommendations_not_found(self, mock_customer_profiles):
        # Mock customer profile data to simulate missing customer
        mock_customer_profiles.__getitem__.return_value = MagicMock()
        mock_customer_profiles["Customer_Id"].unique.return_value = []
        
        response = self.client.get("/content-recommendations/nonexistent")
        
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Customer ID not found")
    
    @patch("app.get_content_recommendations", side_effect=Exception("Unexpected error"))
    @patch("app.customer_profiles")
    def test_get_customer_content_recommendations_server_error(self, mock_customer_profiles, mock_get_content_recommendations):
        # Mock customer profile data
        mock_customer_profiles.__getitem__.return_value = MagicMock()
        mock_customer_profiles["Customer_Id"].unique.return_value = ["CUST2025A"]
        mock_customer_profiles[mock_customer_profiles['Customer_Id'] == "CUST2025A"].iloc.__getitem__.return_value = {
            "Age": 30,
            "Income per year (in dollars)": 75000
        }
        
        response = self.client.get("/content-recommendations/CUST2025A")
        
        self.assertEqual(response.status_code, 500)
        self.assertIn("Error processing request", response.json()["detail"])
    
    #@patch("app.customer_Media", new_callable=lambda: pd.DataFrame)    
    def test_get_social_sentiment_success(self):
        """Test successful sentiment retrieval for an existing customer."""
        #mock_customer_media.return_value = self.mock_customer_media
        
        response = self.client.get("/social-sentiment/CUST2025A")
        self.assertEqual(response.status_code, 200)

    def test_get_social_sentiment_customer_not_found(self):
        """Test case where customer has no social media data (404 error)."""

        response = self.client.get("/social-sentiment/cust999")  # Non-existent ID
        self.assertEqual(response.status_code, 404)
        self.assertIn("No social media data found", response.json()["detail"])
        
    @patch("app.logger.error")
    @patch("app.customer_Media", new_callable=lambda: pd.DataFrame)
    def test_get_social_sentiment_server_error(self, mock_customer_media, mock_logger):
        """Test case where an internal server error occurs (500 error)."""
        mock_customer_media.side_effect = Exception("Database connection lost")  # Simulate DB failure

        response = self.client.get("/social-sentiment/cust123")
        self.assertEqual(response.status_code, 500)
        self.assertIn("Error processing request", response.json()["detail"])
        mock_logger.assert_called_once()
        
if __name__ == "__main__":
    unittest.main()
