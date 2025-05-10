import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class HealthInsightEngine:
    """Core ML/rule-based engine for analyzing health data and generating insights"""
    
    def __init__(self, data_path):
        """Initialize the insight engine with the path to health data"""
        self.data_path = data_path
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load health data from CSV file"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {len(self.data)} records")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None
    
    def preprocess_data(self):
        """Preprocess the health data for analysis"""
        if self.data is None:
            return
        
        # Convert Date to datetime format
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Sort by date
        self.data = self.data.sort_values('Date')
        
        # Fill any missing values (using forward fill for continuity)
        self.data = self.data.fillna(method='ffill')
        
        # Create a numeric representation of mood for easier analysis
        mood_map = {
            'Happy': 5, 
            'Energetic': 4, 
            'Content': 3, 
            'Neutral': 2, 
            'Tired': 1, 
            'Sad': 0,
            'Stressed': 0,
            'Anxious': 0
        }
        
        # Apply mood mapping if the column exists and contains string values
        if 'Mood' in self.data.columns and self.data['Mood'].dtype == 'object':
            self.data['Mood_Score'] = self.data['Mood'].map(lambda x: mood_map.get(x, 2))  # Default to Neutral
    
    def generate_hydration_insights(self):
        """Generate insights related to hydration patterns"""
        insights = []
        
        if 'Hydration (glasses)' not in self.data.columns:
            return ["No hydration data available"]
        
        # Check for decline in hydration over last 3 days
        last_three_days = self.data.tail(3)
        if len(last_three_days) == 3:
            hydration_values = last_three_days['Hydration (glasses)'].values
            if hydration_values[0] > hydration_values[1] > hydration_values[2]:
                insights.append("Your hydration has dropped for 3 consecutive days. Consider increasing water intake.")
            
            # Check if hydration is below recommended level (8 glasses)
            if hydration_values[2] < 8:
                insights.append(f"Your recent hydration ({hydration_values[2]} glasses) is below the recommended 8 glasses per day.")
            
            # Check if hydration has improved
            if hydration_values[0] < hydration_values[1] < hydration_values[2]:
                insights.append("Great job! Your hydration has been improving for the last 3 days.")
        
        # Calculate average hydration
        avg_hydration = self.data['Hydration (glasses)'].mean()
        insights.append(f"Your average daily hydration is {avg_hydration:.1f} glasses of water.")
        
        return insights
    
    def generate_sleep_insights(self):
        """Generate insights related to sleep patterns"""
        insights = []
        
        if 'Sleep (hours)' not in self.data.columns:
            return ["No sleep data available"]
        
        # Check for declining sleep trend
        recent_sleep = self.data.tail(4)['Sleep (hours)'].values
        if len(recent_sleep) >= 3 and all(recent_sleep[i] > recent_sleep[i+1] for i in range(len(recent_sleep)-2)):
            insights.append("Your sleep duration has been decreasing recently. Try to maintain a consistent sleep schedule.")
        
        # Check if sleep is below recommended (7-9 hours)
        recent_avg_sleep = self.data.tail(3)['Sleep (hours)'].mean()
        if recent_avg_sleep < 7:
            insights.append(f"Your recent average sleep ({recent_avg_sleep:.1f} hours) is below the recommended 7-9 hours.")
        elif recent_avg_sleep > 9:
            insights.append(f"Your recent average sleep ({recent_avg_sleep:.1f} hours) is above the typical recommendation. While extra sleep can be beneficial, consistently sleeping more than 9 hours might indicate other health factors.")
        
        # Check for sleep consistency
        sleep_std = self.data['Sleep (hours)'].std()
        if sleep_std > 1.5:
            insights.append(f"Your sleep schedule shows high variability (±{sleep_std:.1f} hours). A consistent sleep schedule may improve your overall energy.")
        
        return insights
    
    def generate_activity_insights(self):
        """Generate insights related to physical activity (steps)"""
        insights = []
        
        if 'Steps' not in self.data.columns:
            return ["No activity data available"]
        
        # Check for recent activity trends
        recent_steps = self.data.tail(4)['Steps'].values
        
        # Check for declining activity
        if len(recent_steps) >= 3 and all(recent_steps[i] > recent_steps[i+1] for i in range(len(recent_steps)-2)):
            insights.append("Your daily step count has been decreasing. Try to incorporate more movement into your routine.")
        
        # Check if activity meets recommended level (10,000 steps)
        recent_avg_steps = self.data.tail(3)['Steps'].mean()
        if recent_avg_steps < 5000:
            insights.append(f"Your recent average of {recent_avg_steps:.0f} steps is below the recommended daily activity level. Aim for at least 10,000 steps per day.")
        elif recent_avg_steps >= 10000:
            insights.append(f"Great job staying active! Your recent average of {recent_avg_steps:.0f} steps meets or exceeds the recommended 10,000 steps per day.")
        
        return insights
    
    def generate_mood_insights(self):
        """Generate insights related to mood patterns"""
        insights = []
        
        if 'Mood' not in self.data.columns:
            return ["No mood data available"]
        
        # Check for consistent negative moods
        if hasattr(self.data, 'Mood_Score'):
            recent_mood_scores = self.data.tail(4)['Mood_Score'].values
            
            if len(recent_mood_scores) >= 3 and np.mean(recent_mood_scores) < 2:
                insights.append("Your recent mood entries suggest you may be experiencing elevated stress or low energy. Consider taking time for self-care activities.")
                
            # Check for mood improvement
            if len(recent_mood_scores) >= 3 and all(recent_mood_scores[i] < recent_mood_scores[i+1] for i in range(len(recent_mood_scores)-2)):
                insights.append("Your mood appears to be improving recently. Keep up the positive momentum!")
        
        # Check for correlation between mood and sleep or hydration
        if all(col in self.data.columns for col in ['Mood_Score', 'Sleep (hours)', 'Hydration (glasses)']):
            sleep_mood_corr = self.data['Mood_Score'].corr(self.data['Sleep (hours)'])
            hydration_mood_corr = self.data['Mood_Score'].corr(self.data['Hydration (glasses)'])
            
            if sleep_mood_corr > 0.4:
                insights.append("There appears to be a positive relationship between your sleep duration and mood. Prioritizing sleep might help maintain a positive outlook.")
                
            if hydration_mood_corr > 0.4:
                insights.append("There seems to be a connection between your hydration levels and mood. Staying hydrated may help support a more positive mood state.")
        
        return insights
    
    def generate_fatigue_insights(self):
        """Generate insights related to fatigue levels"""
        insights = []
        
        if 'Fatigue Level (1-5)' not in self.data.columns:
            return ["No fatigue data available"]
        
        # Check for increasing fatigue trend
        recent_fatigue = self.data.tail(4)['Fatigue Level (1-5)'].values
        if len(recent_fatigue) >= 3 and all(recent_fatigue[i] < recent_fatigue[i+1] for i in range(len(recent_fatigue)-2)):
            insights.append("Your fatigue levels have been increasing. Consider evaluating your rest and recovery strategies.")
        
        # Check if fatigue is consistently high
        recent_avg_fatigue = self.data.tail(3)['Fatigue Level (1-5)'].mean()
        if recent_avg_fatigue > 3.5:
            insights.append(f"Your recent average fatigue level ({recent_avg_fatigue:.1f}/5) is elevated. This could impact your overall wellbeing and productivity.")
        
        # Check for correlation between fatigue and sleep or hydration
        if all(col in self.data.columns for col in ['Fatigue Level (1-5)', 'Sleep (hours)']):
            sleep_fatigue_corr = self.data['Fatigue Level (1-5)'].corr(self.data['Sleep (hours)'])
            
            if sleep_fatigue_corr < -0.3:
                insights.append("There appears to be a relationship between your sleep duration and fatigue levels. Increasing sleep may help reduce fatigue.")
        
        return insights
    
    def get_all_insights(self):
        """Generate comprehensive health insights across all metrics"""
        all_insights = []
        
        all_insights.extend(self.generate_hydration_insights())
        all_insights.extend(self.generate_sleep_insights())
        all_insights.extend(self.generate_activity_insights())
        all_insights.extend(self.generate_mood_insights())
        all_insights.extend(self.generate_fatigue_insights())
        
        return all_insights
    
    def get_weekly_summary(self):
        """Generate a weekly summary of health trends"""
        # Ensure we have at least 7 days of data
        if len(self.data) < 7:
            return "Not enough data for a weekly summary. Please log more days of health data."
        
        last_week = self.data.tail(7)
        
        summary = "Weekly Health Summary:\n"
        
        # Sleep summary
        if 'Sleep (hours)' in last_week.columns:
            avg_sleep = last_week['Sleep (hours)'].mean()
            min_sleep = last_week['Sleep (hours)'].min()
            max_sleep = last_week['Sleep (hours)'].max()
            summary += f"- Sleep: Averaged {avg_sleep:.1f} hours (Range: {min_sleep:.1f}-{max_sleep:.1f} hours)\n"
        
        # Hydration summary
        if 'Hydration (glasses)' in last_week.columns:
            avg_hydration = last_week['Hydration (glasses)'].mean()
            summary += f"- Hydration: Averaged {avg_hydration:.1f} glasses of water per day\n"
        
        # Steps summary
        if 'Steps' in last_week.columns:
            avg_steps = last_week['Steps'].mean()
            summary += f"- Activity: Averaged {avg_steps:.0f} steps per day\n"
        
        # Mood summary
        if 'Mood' in last_week.columns:
            mood_counts = last_week['Mood'].value_counts()
            top_mood = mood_counts.index[0] if not mood_counts.empty else "Unknown"
            summary += f"- Most frequent mood: {top_mood}\n"
        
        # Fatigue summary
        if 'Fatigue Level (1-5)' in last_week.columns:
            avg_fatigue = last_week['Fatigue Level (1-5)'].mean()
            summary += f"- Fatigue Level: Averaged {avg_fatigue:.1f}/5\n"
        
        return summary
    
    def plot_health_trends(self, metric):
        """Generate a plot visualizing trends for a specific health metric"""
        if self.data is None or metric not in self.data.columns:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['Date'], self.data[metric], marker='o', linestyle='-')
        plt.title(f"{metric} Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot to a temporary file
        plot_path = f"health_trend_{metric.replace(' ', '_')}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path


class HealthChatbot:
    """Simple chatbot interface for interacting with health insights"""
    
    def __init__(self, data_path):
        """Initialize the chatbot with the path to health data"""
        self.insight_engine = HealthInsightEngine(data_path)
        
    def process_query(self, query):
        """Process user query and return an appropriate response"""
        query = query.lower().strip()
        
        # Handle general weekly summary
        if re.search(r'how am i doing|weekly|summary|overview', query):
            return self.insight_engine.get_weekly_summary()
        
        # Handle sleep-related queries
        elif re.search(r'sleep|slept|rest', query):
            return self._get_sleep_response(query)
        
        # Handle hydration-related queries
        elif re.search(r'hydration|water|drink|drinking', query):
            return self._get_hydration_response(query)
        
        # Handle activity-related queries
        elif re.search(r'activity|active|steps|walk|walking', query):
            return self._get_activity_response(query)
        
        # Handle mood-related queries
        elif re.search(r'mood|feeling|emotion|happy|sad', query):
            return self._get_mood_response(query)
        
        # Handle fatigue-related queries
        elif re.search(r'fatigue|tired|energy', query):
            return self._get_fatigue_response(query)
        
        # Handle general insights request
        elif re.search(r'insight|recommendation|suggest|advice', query):
            insights = self.insight_engine.get_all_insights()
            return "Health Insights:\n- " + "\n- ".join(insights)
        
        # Handle unknown queries
        else:
            return ("I'm not sure how to answer that. You can ask me about your sleep, hydration, "
                   "activity levels, mood, fatigue, or request a weekly summary or insights.")
    
    def _get_sleep_response(self, query):
        """Generate response for sleep-related queries"""
        sleep_insights = self.insight_engine.generate_sleep_insights()
        
        if re.search(r'trend|pattern', query):
            # Check if we have sleep data
            if 'Sleep (hours)' not in self.insight_engine.data.columns:
                return "No sleep data available to analyze trends."
            
            # Get sleep trend data
            sleep_data = self.insight_engine.data[['Date', 'Sleep (hours)']].tail(14)
            trend_message = "Your recent sleep trend:\n"
            
            for _, row in sleep_data.iterrows():
                trend_message += f"- {row['Date'].strftime('%Y-%m-%d')}: {row['Sleep (hours)']:.1f} hours\n"
            
            # Add insights
            trend_message += "\nAnalysis:\n- " + "\n- ".join(sleep_insights)
            
            return trend_message
        else:
            response = "Sleep Analysis:\n- " + "\n- ".join(sleep_insights)
            
            # Add average sleep info
            if 'Sleep (hours)' in self.insight_engine.data.columns:
                avg_sleep = self.insight_engine.data['Sleep (hours)'].mean()
                response += f"\n\nYour average sleep duration is {avg_sleep:.1f} hours per night."
            
            return response
    
    def _get_hydration_response(self, query):
        """Generate response for hydration-related queries"""
        hydration_insights = self.insight_engine.generate_hydration_insights()
        
        if re.search(r'trend|pattern', query):
            # Check if we have hydration data
            if 'Hydration (glasses)' not in self.insight_engine.data.columns:
                return "No hydration data available to analyze trends."
            
            # Get hydration trend data
            hydration_data = self.insight_engine.data[['Date', 'Hydration (glasses)']].tail(14)
            trend_message = "Your recent hydration trend:\n"
            
            for _, row in hydration_data.iterrows():
                trend_message += f"- {row['Date'].strftime('%Y-%m-%d')}: {row['Hydration (glasses)']:.1f} glasses\n"
            
            # Add insights
            trend_message += "\nAnalysis:\n- " + "\n- ".join(hydration_insights)
            
            return trend_message
        else:
            response = "Hydration Analysis:\n- " + "\n- ".join(hydration_insights)
            
            # Add recommendation
            response += "\n\nRecommendation: Try to drink at least 8 glasses of water each day for optimal hydration."
            
            return response
    
    def _get_activity_response(self, query):
        """Generate response for activity-related queries"""
        activity_insights = self.insight_engine.generate_activity_insights()
        
        if re.search(r'trend|pattern', query):
            # Check if we have activity data
            if 'Steps' not in self.insight_engine.data.columns:
                return "No activity data available to analyze trends."
            
            # Get activity trend data
            step_data = self.insight_engine.data[['Date', 'Steps']].tail(14)
            trend_message = "Your recent activity trend:\n"
            
            for _, row in step_data.iterrows():
                trend_message += f"- {row['Date'].strftime('%Y-%m-%d')}: {row['Steps']:,} steps\n"
            
            # Add insights
            trend_message += "\nAnalysis:\n- " + "\n- ".join(activity_insights)
            
            return trend_message
        else:
            response = "Activity Analysis:\n- " + "\n- ".join(activity_insights)
            
            # Add average step info
            if 'Steps' in self.insight_engine.data.columns:
                avg_steps = self.insight_engine.data['Steps'].mean()
                response += f"\n\nYour average daily activity is {avg_steps:,.0f} steps."
            
            return response
    
    def _get_mood_response(self, query):
        """Generate response for mood-related queries"""
        mood_insights = self.insight_engine.generate_mood_insights()
        
        if re.search(r'trend|pattern', query):
            # Check if we have mood data
            if 'Mood' not in self.insight_engine.data.columns:
                return "No mood data available to analyze trends."
            
            # Get mood trend data
            mood_data = self.insight_engine.data[['Date', 'Mood']].tail(14)
            trend_message = "Your recent mood trend:\n"
            
            for _, row in mood_data.iterrows():
                trend_message += f"- {row['Date'].strftime('%Y-%m-%d')}: {row['Mood']}\n"
            
            # Add insights
            trend_message += "\nAnalysis:\n- " + "\n- ".join(mood_insights)
            
            return trend_message
        else:
            response = "Mood Analysis:\n- " + "\n- ".join(mood_insights)
            
            # Add most frequent mood
            if 'Mood' in self.insight_engine.data.columns:
                most_common_mood = self.insight_engine.data['Mood'].mode()[0]
                response += f"\n\nYour most frequently recorded mood is: {most_common_mood}"
            
            return response
    
    def _get_fatigue_response(self, query):
        """Generate response for fatigue-related queries"""
        fatigue_insights = self.insight_engine.generate_fatigue_insights()
        
        if re.search(r'trend|pattern', query):
            # Check if we have fatigue data
            if 'Fatigue Level (1-5)' not in self.insight_engine.data.columns:
                return "No fatigue data available to analyze trends."
            
            # Get fatigue trend data
            fatigue_data = self.insight_engine.data[['Date', 'Fatigue Level (1-5)']].tail(14)
            trend_message = "Your recent fatigue trend:\n"
            
            for _, row in fatigue_data.iterrows():
                trend_message += f"- {row['Date'].strftime('%Y-%m-%d')}: Level {row['Fatigue Level (1-5)']}/5\n"
            
            # Add insights
            trend_message += "\nAnalysis:\n- " + "\n- ".join(fatigue_insights)
            
            return trend_message
        else:
            response = "Fatigue Analysis:\n- " + "\n- ".join(fatigue_insights)
            
            # Add average fatigue level
            if 'Fatigue Level (1-5)' in self.insight_engine.data.columns:
                avg_fatigue = self.insight_engine.data['Fatigue Level (1-5)'].mean()
                response += f"\n\nYour average fatigue level is {avg_fatigue:.1f}/5."
            
            return response


def main():
    """Main function to run the Health Insight Chatbot"""
    print("Welcome to Sparkle Health Insight Chatbot!")
    print("-------------------------------------------")
    
    # Use the provided CSV file path or default to a sample path
    
    # Initialize the chatbot
    
    print(f"Data loaded from:")
    print("You can ask me questions about your health trends and insights.")
    print("Type 'exit' to quit.")
    print("-------------------------------------------")
    
    # Print a general overview to start
    print("Here's a quick overview of your recent health data:")
    print(chatbot.insight_engine.get_weekly_summary())
    print("-------------------------------------------")
    
    # Interactive loop
    while True:
        user_input = input("\nWhat would you like to know about your health? ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using Sparkle Health Insight Chatbot. Stay healthy!")
            break
        
        response = chatbot.process_query(user_input)
        print("\n" + response)


import streamlit as st
from io import StringIO

st.set_page_config(page_title="Health Insight Dashboard", layout="wide")
st.title("🧠 Personal Health Insight System")
st.markdown("Upload your health data CSV file to get personalized wellness feedback.")

uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the file as a string buffer
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    # Save temporarily to pass to your class
    temp_path = "temp_health_data.csv"
    with open(temp_path, "w") as f:
        f.write(stringio.getvalue())

    # Initialize engine
    engine = HealthInsightEngine(data_path=temp_path)
    chatbot = HealthChatbot(data_path=temp_path)
    
    # Display raw data
    st.subheader("📊 Uploaded Health Data")
    st.dataframe(engine.data)
    
    st.subheader("🧠 Ask Your Health Assistant")
    user_query = st.text_input("Type your question (e.g., 'How is my sleep this week?')")

    if user_query:
        response = chatbot.process_query(user_query)
        st.markdown(f"**🤖 Response:** {response}")


else:
    st.info("Please upload a CSV file to get started.")
