# seed_database.py

import pandas as pd
from sqlalchemy import create_engine
import os

# --- Database Connection ---
DB_USER = os.environ.get("DB_USER", "postgres.doexzugluvzzshbkduyn")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "FIFAUKEPL2025")
DB_HOST = os.environ.get("DB_HOST", "aws-0-us-east-1.pooler.supabase.com")
DB_PORT = os.environ.get("DB_PORT", "6543")
DB_NAME = os.environ.get("DB_NAME", "postgres")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def seed_epl_data():
    """Reads epl_final.csv and loads it into the epl_match table."""
    file_path = os.path.join('attached_assets', 'epl_final.csv')
    try:
        df = pd.read_csv(file_path)
        print(f"Read {len(df)} rows from epl_final.csv")

        column_mapping = {
            'Season': 'season',
            'MatchDate': 'match_date',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'FullTimeHomeGoals': 'full_time_home_goals',
            'FullTimeAwayGoals': 'full_time_away_goals',
            'FullTimeResult': 'full_time_result',
            'HalfTimeHomeGoals': 'half_time_home_goals',
            'HalfTimeAwayGoals': 'half_time_away_goals',
            'HalfTimeResult': 'half_time_result',
            'HomeShots': 'home_shots',
            'AwayShots': 'away_shots',
            'HomeShotsOnTarget': 'home_shots_on_target',
            'AwayShotsOnTarget': 'away_shots_on_target',
            'HomeCorners': 'home_corners',
            'AwayCorners': 'away_corners',
            'HomeFouls': 'home_fouls',
            'AwayFouls': 'away_fouls',
            'HomeYellowCards': 'home_yellow_cards',
            'AwayYellowCards': 'away_yellow_cards',
            'HomeRedCards': 'home_red_cards',
            'AwayRedCards': 'away_red_cards'
        }
        df.rename(columns=column_mapping, inplace=True)
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')

        db_columns = list(column_mapping.values())
        df_to_load = df[db_columns]
        df_to_load.to_sql('epl_match', engine, if_exists='replace', index=False)

        print("✅ Successfully seeded the epl_match table.")
    except Exception as e:
        print(f"❌ Error seeding EPL data: {e}")

def seed_ucl_data():
    """Reads ucl_Matches.csv and loads it into the ucl_matches table."""
    file_path = os.path.join('attached_assets', 'ucl_Matches.csv')
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Read {len(df)} rows from ucl_Matches.csv")

        column_mapping = {
            'Division': 'stage',
            'MatchDate': 'match_date',
            'MatchTime': 'match_time',
            'HomeTeam': 'home_team',
            'AwayTeam': 'away_team',
            'HomeElo': 'home_elo',
            'AwayElo': 'away_elo',
            'Form3Home': 'form3_home',
            'Form5Home': 'form5_home',
            'Form3Away': 'form3_away',
            'Form5Away': 'form5_away',
            'FTHome': 'ft_home',
            'FTAway': 'ft_away',
            'FTResult': 'ft_result',
            'HTHome': 'ht_home',
            'HTAway': 'ht_away',
            'HTResult': 'ht_result',
            'HomeShots': 'home_shots',
            'AwayShots': 'away_shots',
            'HomeTarget': 'home_target',
            'AwayTarget': 'away_target',
            'HomeFouls': 'home_fouls',
            'AwayFouls': 'away_fouls',
            'HomeCorners': 'home_corners',
            'AwayCorners': 'away_corners',
            'HomeYellow': 'home_yellow',
            'AwayYellow': 'away_yellow',
            'HomeRed': 'home_red',
            'AwayRed': 'away_red',
            'OddHome': 'odd_home',
            'OddDraw': 'odd_draw',
            'OddAway': 'odd_away',
            'MaxHome': 'max_home',
            'MaxDraw': 'max_draw',
            'MaxAway': 'max_away',
            'Over25': 'over_25',
            'Under25': 'under_25',
            'MaxOver25': 'max_over_25',
            'MaxUnder25': 'max_under_25',
            'HandiSize': 'handi_size',
            'HandiHome': 'handi_home',
            'HandiAway': 'handi_away'
        }
        df.rename(columns=column_mapping, inplace=True)

        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        df['season'] = df['match_date'].dt.year.apply(lambda x: f"{x-1}/{str(x)[-2:]}" if pd.notna(x) else None)

        db_columns = ['season', 'match_date', 'stage', 'home_team', 'away_team', 'ft_home', 'ft_away', 'home_shots', 'away_shots', 'home_target', 'away_target', 'home_fouls', 'away_fouls', 'home_corners', 'away_corners', 'home_yellow', 'away_yellow', 'home_red', 'away_red', 'ft_result', 'ht_home', 'ht_away', 'ht_result', 'home_elo', 'away_elo', 'form3_home', 'form5_home', 'form3_away', 'form5_away', 'odd_home', 'odd_draw', 'odd_away', 'max_home', 'max_draw', 'max_away', 'over_25', 'under_25', 'max_over_25', 'max_under_25', 'handi_size', 'handi_home', 'handi_away', 'match_time']
        df_to_load = df[db_columns]
        df_to_load.to_sql('ucl_matches', engine, if_exists='replace', index=False)

        print("✅ Successfully seeded the ucl_matches table.")
    except Exception as e:
        print(f"❌ Error seeding UCL data: {e}")

if __name__ == "__main__":
    print("🚀 Starting Database Seeding")
    seed_epl_data()
    seed_ucl_data()
    print("✅ All Seeding Complete")
