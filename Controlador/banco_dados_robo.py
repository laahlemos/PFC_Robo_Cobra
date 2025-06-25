import mysql.connector

def connect_to_db():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="lara",
        password="xxx",
        database="simulation_db"
    )

def setup_database():
    connection = connect_to_db()
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS base_resultados (
            id INT AUTO_INCREMENT PRIMARY KEY,
            parameters TEXT NOT NULL,
            result TEXT NOT NULL,
            initial_position VARCHAR(255),
            final_position VARCHAR(255),
            displacement_vector VARCHAR(255)
        )
    ''')

    connection.commit()
    connection.close()

def check_parameters(parameters):
    connection = connect_to_db()
    cursor = connection.cursor()

    query = 'SELECT * FROM base_resultados WHERE parameters = %s'
    cursor.execute(query, (parameters,))
    exists = cursor.fetchone() is not None

    connection.close()
    return exists

def save_test_result(parameters, result, initial_position, final_position, displacement_vector):
    connection = connect_to_db()
    cursor = connection.cursor()

    query = '''
        INSERT INTO base_resultados (parameters, result, initial_position, final_position, displacement_vector)
        VALUES (%s, %s, %s, %s, %s)
    '''
    cursor.execute(query, (parameters, result, initial_position, final_position, displacement_vector))

    connection.commit()
    connection.close()

if __name__ == "__main__":
    setup_database()  
