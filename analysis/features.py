# Determine response times between messages

import pandas as pd
data_dir = '../data/'

def create_response_times():
    messages = pd.read_csv('messages.csv')
    messages = messages.sort_values(['user_id', 'session_type', 'message_index'])

    # Shift to get the previous message's role and timestamp
    messages['prev_role'] = messages.groupby(['user_id', 'session_type'])['role'].shift(1)
    messages['prev_timestamp'] = messages.groupby(['user_id', 'session_type'])['timestamp'].shift(1)

    # Calculate time since previous message
    messages['time_since_prev'] = messages['timestamp'] - messages['prev_timestamp']

    # Filter: only user messages that follow an assistant message
    user_responses = messages[
        (messages['role'] == 'user') &
        (messages['prev_role'] == 'assistant')
        ]

    user_responses['quick_response'] = user_responses['time_since_prev'] < 10
    quick_count = user_responses.groupby(['user_id', 'session_type'])['quick_response'].sum().reset_index()
    quick_count.columns = ['user_id', 'session_type', 'rapid_response_count']

    # Aggregate per session
    response_times = user_responses.groupby(['user_id', 'session_type']).agg(
        avg_response_time=('time_since_prev', 'mean'),
        median_response_time=('time_since_prev', 'median'),
        std_response_time=('time_since_prev', 'std'),
        min_response_time=('time_since_prev', 'min'),
        max_response_time=('time_since_prev', 'max')
    ).reset_index()

    return response_times




def merge_to_sessions():
    sessions = pd.read_csv(data_dir + 'sessions.csv')
    session_pacing = create_response_times()
    merged = pd.merge(sessions, session_pacing, on=['user_id', 'session_type'])
    return merged



