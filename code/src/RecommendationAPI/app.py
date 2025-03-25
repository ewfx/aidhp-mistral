#Fast APi Packages
from fastapi import FastAPI, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from datetime import datetime, timedelta
from statistics import mean
import warnings
import os
import logging
import requests
import io
import os
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from dotenv import load_dotenv
 # load all the environment variables
from openai import OpenAI
warnings.filterwarnings('ignore')

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVidea_Key")
)

load_dotenv() 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# URL of the Excel file
EXCEL_URL = "https://huggingface.co/spaces/Vaibhav84/RecommendationAPI/resolve/main/DataSetSample.xlsx"

try:
    # Download the file from URL
    logger.info(f"Attempting to download Excel file from: {EXCEL_URL}")
    response = requests.get(EXCEL_URL)
    response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
    
    # Read the Excel file from the downloaded content
    excel_content = io.BytesIO(response.content)
    purchase_history = pd.read_excel(excel_content, sheet_name='Transaction History', 
                                   parse_dates=['Purchase_Date'])
    
    # Read Customer Profile sheet
    excel_content.seek(0)  # Reset buffer position
    customer_profiles = pd.read_excel(excel_content, sheet_name='Customer Profile (Individual)')

    # Read Social Media Sentiment
    excel_content.seek(0)  # Reset buffer position
    customer_Media = pd.read_excel(excel_content, sheet_name='Social Media Sentiment',parse_dates=['Timestamp'])

    logger.info("Successfully downloaded and loaded Excel file")
    
    # Process the data
    purchase_history['Customer_Id'] = purchase_history['Customer_Id'].astype(str)
    product_categories = purchase_history[['Product_Id', 'Category']].drop_duplicates().set_index('Product_Id')['Category'].to_dict()
    purchase_counts = purchase_history.groupby(['Customer_Id', 'Product_Id']).size().unstack(fill_value=0)
    sparse_purchase_counts = sparse.csr_matrix(purchase_counts)
    cosine_similarities = cosine_similarity(sparse_purchase_counts.T)

     # Process customer profiles data
    customer_profiles['Customer_Id'] = customer_profiles['Customer_Id'].astype(str)
    
    # Normalize numerical features if they exist
    numerical_features = ['Age', 'Income per year (in dollars)']  # Add or modify based on your actual columns
    scaler = StandardScaler()
    customer_profiles[numerical_features] = scaler.fit_transform(customer_profiles[numerical_features])

      # Process the data media
    customer_Media['Customer_Id'] = customer_Media['Customer_Id'].astype(str)
    tweet_categories = customer_Media[['Post_Id', 'Platform']].drop_duplicates().set_index('Post_Id')['Platform'].to_dict()
    tweet_counts = customer_Media.groupby(['Customer_Id', 'Post_Id']).size().unstack(fill_value=0)
    sparse_tweet_counts = sparse.csr_matrix(tweet_counts)
    cosine_similarities_tweet = cosine_similarity(sparse_tweet_counts.T)
    
    logger.info("Data processing completed successfully")
    
except Exception as e:
    logger.error(f"Error downloading or processing data: {str(e)}")
    raise

def get_customer_items_and_recommendations(user_id: str, n: int = 5) -> tuple[List[Dict], List[Dict]]:
    """
    Get both purchased items and recommendations for a user
    """
    user_id = str(user_id)
    
    if user_id not in purchase_counts.index:
        return [], []
    
    purchased_items = list(purchase_counts.columns[purchase_counts.loc[user_id] > 0])
    
    purchased_items_info = []
    user_purchases = purchase_history[purchase_history['Customer_Id'] == user_id]
    
    for item in purchased_items:
        item_purchases = user_purchases[user_purchases['Product_Id'] == item]
        total_amount = float(item_purchases['Amount (In Dollars)'].sum())
        last_purchase = pd.to_datetime(item_purchases['Purchase_Date'].max())
        category = product_categories.get(item, 'Unknown')
        purchased_items_info.append({
            'product_id': item,
            'category': category,
            'total_amount': total_amount,
            'last_purchase': last_purchase.strftime('%Y-%m-%d')
        })
    
    user_idx = purchase_counts.index.get_loc(user_id)
    user_history = sparse_purchase_counts[user_idx].toarray().flatten()
    similarities = cosine_similarities.dot(user_history)
    purchased_indices = np.where(user_history > 0)[0]
    similarities[purchased_indices] = 0
    recommended_indices = np.argsort(similarities)[::-1][:n]
    recommended_items = list(purchase_counts.columns[recommended_indices])
    recommended_items = [item for item in recommended_items if item not in purchased_items]
    
    recommended_items_info = [
        {
            'product_id': item,
            'category': product_categories.get(item, 'Unknown')
        }
        for item in recommended_items
    ]

    return purchased_items_info, recommended_items_info

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Recommendation API",
        "status": "running",
        "data_loaded": purchase_history is not None
    }

@app.get("/recommendations/{customer_id}")
async def get_recommendations(customer_id: str, n: int = 5):
    """
    Get recommendations for a customer
    
    Parameters:
    - customer_id: The ID of the customer
    - n: Number of recommendations to return (default: 5)
    
    Returns:
    - JSON object containing purchase history and recommendations
    """
    try:
        purchased_items, recommended_items = get_customer_items_and_recommendations(customer_id, n)
        
        return {
            "customer_id": customer_id,
            "purchase_history": purchased_items,
            "recommendations": recommended_items
        }
    except Exception as e:
        logger.error(f"Error processing request for customer {customer_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Error processing customer ID: {customer_id}. {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint that returns system information
    """
    return {
        "status": "healthy",
        "data_loaded": purchase_history is not None,
        "number_of_customers": len(purchase_counts.index) if purchase_history is not None else 0,
        "number_of_products": len(purchase_counts.columns) if purchase_history is not None else 0
    }

@app.post("/login")
async def login(customer_id: str, password: str):
    """
    Login endpoint to validate customer ID and password
    
    Parameters:
    - customer_id: The ID of the customer to validate
    - password: Password (first three chars of customer_id + "123")
    
    Returns:
    - JSON object containing login status and message
    """
    try:
        # Convert customer_id to string to match the format in purchase_history
        customer_id = str(customer_id)
        
        # Generate expected password (first three chars + "123")
        expected_password = f"{customer_id[:3]}123"
        
        # Check if customer exists and password matches
        if customer_id in purchase_history['Customer_Id'].unique():
            if password == expected_password:
                # Get customer's basic information
                customer_data = purchase_history[purchase_history['Customer_Id'] == customer_id]
                total_purchases = len(customer_data)
                total_spent = customer_data['Amount (In Dollars)'].sum()
                
                # Convert last purchase date to datetime if it's not already
                last_purchase = pd.to_datetime(customer_data['Purchase_Date'].max())
                last_purchase_str = last_purchase.strftime('%Y-%m-%d')
                
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "status": "success",
                        "message": "Login successful",
                        "customer_id": customer_id,
                        "customer_stats": {
                            "total_purchases": total_purchases,
                            "total_spent": float(total_spent),
                            "last_purchase_date": last_purchase_str
                        }
                    }
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "status": "error",
                        "message": "Invalid password"
                    }
                )
        else:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "status": "error",
                    "message": "Invalid customer ID"
                }
            )
            
    except Exception as e:
        logger.error(f"Error during login for customer {customer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during login process: {str(e)}"
        )
# Add content recommendation function
def get_content_recommendations(customer_id: str, n: int = 5) -> List[Dict]:
    """
    Get content recommendations based on customer profile
    """
    try:
        # Get customer profile
        customer_profile = customer_profiles[customer_profiles['Customer_Id'] == customer_id].iloc[0]
        
        # Define content rules based on customer attributes
        content_suggestions = []
        
        # Age-based recommendations
        age = customer_profile['Age'] * scaler.scale_[0] + scaler.mean_[0]  # Denormalize age
        
        if age < 25:
            content_suggestions.extend([
                {"type": "Video", "title": "Getting Started with Personal Finance", "category": "Financial Education"},
                {"type": "Article", "title": "Budgeting Basics for Young Adults", "category": "Financial Planning"},
                {"type": "Interactive", "title": "Investment 101 Quiz", "category": "Education"}
            ])
        elif age < 40:
            content_suggestions.extend([
                {"type": "Video", "title": "Investment Strategies for Growing Wealth", "category": "Investment"},
                {"type": "Article", "title": "Family Financial Planning Guide", "category": "Financial Planning"},
                {"type": "Webinar", "title": "Real Estate Investment Basics", "category": "Investment"}
            ])
        else:
            content_suggestions.extend([
                {"type": "Video", "title": "Retirement Planning Strategies", "category": "Retirement"},
                {"type": "Article", "title": "Estate Planning Essentials", "category": "Financial Planning"},
                {"type": "Webinar", "title": "Tax Optimization for Retirement", "category": "Tax Planning"}
            ])
        
        # Income-based recommendations
        income = customer_profile['Income per year (in dollars)'] * scaler.scale_[1] + scaler.mean_[1]  # Denormalize income
        
        if income < 50000:
            content_suggestions.extend([
                {"type": "Video", "title": "Debt Management Strategies", "category": "Debt Management"},
                {"type": "Article", "title": "Saving on a Tight Budget", "category": "Budgeting"}
            ])
        elif income < 100000:
            content_suggestions.extend([
                {"type": "Video", "title": "Tax-Efficient Investment Strategies", "category": "Investment"},
                {"type": "Article", "title": "Maximizing Your 401(k)", "category": "Retirement"}
            ])
        else:
            content_suggestions.extend([
                {"type": "Video", "title": "Advanced Tax Planning Strategies", "category": "Tax Planning"},
                {"type": "Article", "title": "High-Net-Worth Investment Guide", "category": "Investment"}
            ])
        
        # Add personalization based on purchase history
        if customer_id in purchase_history['Customer_Id'].unique():
            customer_purchases = purchase_history[purchase_history['Customer_Id'] == customer_id]
            categories = customer_purchases['Category'].unique()
            
            for category in categories:
                if category == 'Investment':
                    content_suggestions.append({
                        "type": "Video",
                        "title": f"Advanced {category} Strategies",
                        "category": category
                    })
                elif category == 'Insurance':
                    content_suggestions.append({
                        "type": "Article",
                        "title": f"Understanding Your {category} Options",
                        "category": category
                    })
        
        # Remove duplicates and limit to n recommendations
        seen = set()
        unique_suggestions = []
        for suggestion in content_suggestions:
            key = (suggestion['title'], suggestion['type'])
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:n]
        
    except Exception as e:
        logger.error(f"Error generating content recommendations: {str(e)}")
        return []

# Add new endpoint for content recommendations
@app.get("/content-recommendations/{customer_id}")
async def get_customer_content_recommendations(customer_id: str, n: int = 5):
    """
    Get personalized content recommendations for a customer
    
    Parameters:
    - customer_id: The ID of the customer
    - n: Number of recommendations to return (default: 5)
    
    Returns:
    - JSON object containing personalized content recommendations
    """
    try:
        # Validate customer
        if customer_id not in customer_profiles['Customer_Id'].unique():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer ID not found"
            )
        
        # Get customer profile summary
        customer_profile = customer_profiles[customer_profiles['Customer_Id'] == customer_id].iloc[0]
        profile_summary = {
            "age_group": "Young" if customer_profile['Age'] < 25 else "Middle" if customer_profile['Age'] < 40 else "Senior",
            "income_level": "Low" if customer_profile['Income per year (in dollars)'] < 50000 else "Medium" if customer_profile['Income per year (in dollars)'] < 100000 else "High"
        }
        
        # Get content recommendations
        recommendations = get_content_recommendations(customer_id, n)
        
        return {
            "customer_id": customer_id,
            "profile_summary": profile_summary,
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing content recommendations for customer {customer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )
    
@app.get("/social-sentiment/{customer_id}")
async def get_social_sentiment(customer_id: str):
    """
    Get social media sentiment analysis for a customer
    
    Parameters:
    - customer_id: The ID of the customer
    
    Returns:
    - JSON object containing sentiment analysis and insights
    """
    try:
        # Validate customer
        if customer_id not in customer_Media['Customer_Id'].unique():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No social media data found for this customer"
            )
        
        # Get customer's social media data
        customer_posts = customer_Media[customer_Media['Customer_Id'] == customer_id]
        
        # Calculate sentiment metrics
        avg_sentiment = customer_posts['Sentiment_Score'].mean()
        recent_sentiment = customer_posts.sort_values('Timestamp', ascending=False)['Sentiment_Score'].iloc[0]
        
        # Calculate sentiment trend
        customer_posts['Timestamp'] = pd.to_datetime(customer_posts['Timestamp'])
        sentiment_trend = customer_posts.sort_values('Timestamp')
        
        # Platform breakdown
        platform_stats = customer_posts.groupby('Platform').agg({
            'Post_Id': 'count',
            'Sentiment_Score': 'mean'
        }).round(2)
        
        platform_breakdown = [
            {
                "platform": platform,
                "post_count": int(stats['Post_Id']),
                "avg_sentiment": float(stats['Sentiment_Score'])
            }
            for platform, stats in platform_stats.iterrows()
        ]
        
        # Intent analysis
        intent_distribution = customer_posts['Intent'].value_counts().to_dict()
        
        # Get most recent posts with sentiments
        recent_posts = customer_posts.sort_values('Timestamp', ascending=False).head(5)
        recent_activities = [
            {
                "timestamp": post['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "platform": post['Platform'],
                "content": post['Content'],
                "sentiment_score": float(post['Sentiment_Score']),
                "intent": post['Intent']
            }
            for _, post in recent_posts.iterrows()
        ]
        
        # Calculate sentiment categories
        sentiment_categories = {
            "positive": len(customer_posts[customer_posts['Sentiment_Score'] > 0.5]),
            "neutral": len(customer_posts[(customer_posts['Sentiment_Score'] >= -0.5) & 
                                       (customer_posts['Sentiment_Score'] <= 0.5)]),
            "negative": len(customer_posts[customer_posts['Sentiment_Score'] < -0.5])
        }
        
        # Determine overall mood
        if avg_sentiment > 0.5:
            overall_mood = "Positive"
        elif avg_sentiment < -0.5:
            overall_mood = "Negative"
        else:
            overall_mood = "Neutral"
            
        # Generate insights
        insights = []
        
        # Trend insight
        sentiment_change = recent_sentiment - customer_posts['Sentiment_Score'].iloc[0]
        if abs(sentiment_change) > 0.3:
            trend_direction = "improved" if sentiment_change > 0 else "declined"
            insights.append(f"Customer sentiment has {trend_direction} over time")
            
        # Platform insight
        if len(platform_stats) > 1:
            best_platform = platform_stats['Sentiment_Score'].idxmax()
            insights.append(f"Customer shows most positive engagement on {best_platform}")
            
        # Engagement insight
        if len(recent_activities) > 0:
            recent_avg = sum(post['sentiment_score'] for post in recent_activities) / len(recent_activities)
            if abs(recent_avg - avg_sentiment) > 0.3:
                trend = "improving" if recent_avg > avg_sentiment else "declining"
                insights.append(f"Recent sentiment is {trend} compared to overall average")
        
        return {
            "customer_id": customer_id,
            "overall_sentiment": {
                "average_score": float(avg_sentiment),
                "recent_score": float(recent_sentiment),
                "overall_mood": overall_mood
            },
            "sentiment_distribution": sentiment_categories,
            "platform_analysis": platform_breakdown,
            "intent_analysis": intent_distribution,
            "recent_activities": recent_activities,
            "insights": insights,
            "analysis_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing social sentiment for customer {customer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

# Add a combined endpoint for full customer analysis
@app.get("/customer-analysis/{customer_id}")
async def get_customer_analysis(customer_id: str):
    """
    Get comprehensive customer analysis including recommendations and sentiment
    
    Parameters:
    - customer_id: The ID of the customer
    
    Returns:
    - JSON object containing full customer analysis
    """
    try:
        # Get content recommendations
        content_recs = await get_customer_content_recommendations(customer_id)
        
        # Get social sentiment
        sentiment_analysis = await get_social_sentiment(customer_id)
        
        # Get purchase recommendations
        purchase_recs = await get_recommendations(customer_id)
        
        return {
            "customer_id": customer_id,
            "sentiment_analysis": sentiment_analysis,
            "content_recommendations": content_recs,
            "purchase_recommendations": purchase_recs,
            "analysis_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Error processing customer analysis for {customer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )
@app.get("/financial-recommendations/{customer_id}")
async def get_financial_recommendations(customer_id: str):
    """
    Get hyper-personalized financial recommendations for a customer
    """
    try:
        # Validate customer
        if customer_id not in customer_profiles['Customer_Id'].values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Get customer profile data
        customer_profile = customer_profiles[customer_profiles['Customer_Id'] == customer_id].iloc[0]
        
        # Get purchase history
        customer_purchases = purchase_history[purchase_history['Customer_Id'] == customer_id]
        
        # Get social sentiment data
        customer_sentiment = customer_Media[customer_Media['Customer_Id'] == customer_id]
        
        # Calculate financial metrics with type conversion
        try:
            total_spent = customer_purchases['Amount (In Dollars)'].astype(float).sum()
            avg_transaction = customer_purchases['Amount (In Dollars)'].astype(float).mean()
            
            # Convert purchase dates to datetime
            customer_purchases['Purchase_Date'] = pd.to_datetime(customer_purchases['Purchase_Date'])
            date_range = (customer_purchases['Purchase_Date'].max() - customer_purchases['Purchase_Date'].min()).days
            purchase_frequency = len(customer_purchases) / (date_range + 1) if date_range > 0 else 0
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing numerical calculations: {str(e)}")
            total_spent = 0
            avg_transaction = 0
            purchase_frequency = 0
        
        try:
            # Convert age and income to float
            age = float(customer_profile['Age'])
            income = float(customer_profile['Income per year (in dollars)'])
            
            # Calculate spending ratio
            spending_ratio = (total_spent / income) * 100 if income > 0 else 0
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing profile data: {str(e)}")
            age = 0
            income = 0
            spending_ratio = 0
        
        # Generate recommendations based on processed data
        recommendations = {
            "investment_recommendations": [],
            "savings_recommendations": [],
            "budget_recommendations": [],
            "risk_assessment": "",
            "action_items": []
        }
        
        # Investment recommendations based on age
        if age < 30:
            recommendations["investment_recommendations"] = [
                "Consider starting a retirement account with aggressive growth funds",
                "Look into low-cost index funds for long-term growth",
                "Build an emergency fund of 3-6 months expenses"
            ]
        elif age < 50:
            recommendations["investment_recommendations"] = [
                "Diversify investment portfolio with mix of stocks and bonds",
                "Consider real estate investment opportunities",
                "Maximize retirement contributions"
            ]
        else:
            recommendations["investment_recommendations"] = [
                "Focus on preservation of capital",
                "Consider dividend-paying stocks",
                "Review retirement withdrawal strategy"
            ]
        
        # Savings recommendations based on spending ratio
        if spending_ratio > 70:
            recommendations["savings_recommendations"] = [
                "Critical: Reduce monthly expenses",
                "Implement 50/30/20 budgeting rule",
                "Identify and cut non-essential spending"
            ]
        elif spending_ratio > 50:
            recommendations["savings_recommendations"] = [
                "Look for additional saving opportunities",
                "Consider automated savings transfers",
                "Review subscription services"
            ]
        else:
            recommendations["savings_recommendations"] = [
                "Maintain current saving habits",
                "Consider increasing investment contributions",
                "Look into tax-advantaged savings options"
            ]
        
        # Budget recommendations based on purchase patterns
        try:
            category_spending = customer_purchases.groupby('Category')['Amount (In Dollars)'].astype(float).sum()
            top_spending_categories = category_spending.nlargest(3)
            
            recommendations["budget_recommendations"] = [
                f"Highest spending in {cat}: ${amount:.2f}" 
                for cat, amount in top_spending_categories.items()
            ]
        except Exception as e:
            logger.error(f"Error processing category spending: {str(e)}")
            recommendations["budget_recommendations"] = ["Unable to process category spending"]
        
        # Risk assessment based on sentiment
        try:
            recent_sentiment = customer_sentiment['Sentiment_Score'].astype(float).mean()
            if pd.isna(recent_sentiment):
                risk_level = "Balanced"
            elif recent_sentiment < -0.2:
                risk_level = "Conservative"
            elif recent_sentiment > 0.2:
                risk_level = "Moderate"
            else:
                risk_level = "Balanced"
        except Exception as e:
            logger.error(f"Error processing sentiment: {str(e)}")
            risk_level = "Balanced"
            
        recommendations["risk_assessment"] = f"Based on your profile and behavior: {risk_level} risk tolerance"
        
        # Action items
        recommendations["action_items"] = [
            {
                "priority": "High",
                "action": "Review and adjust monthly budget",
                "impact": "Immediate",
                "timeline": "Next 30 days"
            },
            {
                "priority": "Medium",
                "action": "Rebalance investment portfolio",
                "impact": "Long-term",
                "timeline": "Next 90 days"
            },
            {
                "priority": "Low",
                "action": "Schedule financial planning review",
                "impact": "Strategic",
                "timeline": "Next 6 months"
            }
        ]
        
        return {
            "customer_id": customer_id,
            "financial_summary": {
                "total_spent": float(total_spent),
                "average_transaction": float(avg_transaction),
                "spending_ratio": float(spending_ratio),
                "purchase_frequency": float(purchase_frequency)
            },
            "risk_profile": {
                "age_group": "Young" if age < 30 else "Middle-aged" if age < 50 else "Senior",
                "income_bracket": "Low" if income < 50000 else "Medium" if income < 100000 else "High",
                "risk_tolerance": risk_level
            },
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Error processing financial recommendations for customer {customer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

def Get_contentDescription(category:str):
    inputdata = "Generate articles for "
    inputdata += category +" in few words. Don't show the thinking part in the output"
    completion = client.chat.completions.create(
    model="microsoft/phi-4-mini-instruct",
    messages=[{"role":"user","content":inputdata}],
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
    stream=True
    )
    full_response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
        # Clean the complete response
    cleaned_response = full_response.replace("<think>", "").replace("</think>", "").strip()
    return(cleaned_response)
@app.get("/contentcreation/{category}")
async def contentcreation(category:str):
    cleaned_response= Get_contentDescription(category)
    return(cleaned_response)
