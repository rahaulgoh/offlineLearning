import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import time
import logging

# Config
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "mt10ma18",
    "host": "192.168.0.86",
    "port": "5432",
}

RAW_TABLES = [
    'raw_temp_record',
    'raw_vibration_record',
    'raw_humid_record',
    'raw_motion_record',
    'raw_accel_record'
]

LOOKBACK_MINUTES = 5

RUN_INTERVAL_SECONDS = 10

# Setup logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_best_gateway_per_tag(conn, lookback_minutes):
    cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)

    union_parts = []
    for table in RAW_TABLES:
        union_parts.append(f"""
            SELECT tag_id, gateway_id, rssi, created_on
            FROM {table}
            WHERE created_on >= %s
        """)

    union_query = " UNION ALL ".join(union_parts)
    
    query = f"""
    WITH all_readings AS (
        {union_query}
    ),
    ranked AS (
        SELECT
            tag_id,
            gateway_id,
            rssi,
            ROW_NUMBER() OVER(
                PARTITION BY tag_id
                ORDER BY rssi DESC, created_on DESC
            ) as rn
        FROM all_readings
    )
    SELECT tag_id, gateway_id
    FROM ranked
    WHERE rn = 1
    """

    params = [cutoff_time] * len(RAW_TABLES)

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()
    
def update_assignments(conn, assignments):
    if not assignments:
        logger.info("No assignments to update.")
        return 0
    
    upsert_sql = """
    INSERT INTO tag_gateway_assignment (tag_id, assigned_gateway_id, updated_on)
    VALUES %s
    ON CONFLICT (tag_id) DO UPDATE SET
        assigned_gateway_id = EXCLUDED.assigned_gateway_id,
        updated_on = EXCLUDED.updated_on;
    """

    now = datetime.now()
    data = [(tag_id, gateway_id, now) for tag_id, gateway_id in assignments]

    with conn.cursor() as cur:
        execute_values(cur, upsert_sql, data)
    conn.commit()
    return len(data)


def main():
    logger.info("Starting tag-gateway assignment service")

    while True:
        try:
            conn = get_connection()
            assignments = get_best_gateway_per_tag(conn, LOOKBACK_MINUTES)
            updated = update_assignments(conn, assignments)
            logger.info(f"Updated assignments for {updated} tags.")
            conn.close()
        except Exception as e:
            logger.error(f"Error during processing: {e}")

        time.sleep(RUN_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()