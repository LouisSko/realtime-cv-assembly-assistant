import pytest
from app import app, LABELS, STEPS


# Create a test client for the app
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

########################## Test cases for the HTTP requests ##########################

def test_index_page(client):
    # Test if the index page loads successfully
    response = client.get('/')
    assert response.status_code == 200
    assert b"LEGO Mindstorm: Real-Time Instruction Manual" in response.data

def test_not_found_page(client):
    # Test if accessing a non-existent page returns status code 404
    response = client.get('/non-existent-page')
    assert response.status_code == 404
    assert b"404 Not Found" in response.data

def test_set_settings(client):
    # Test if setting settings returns a JSON response with status code 200
    data = {
        'coloring': 'single-color',
        'confidence': '50',
        'displayConfidence': '',
        'displayLabel': '',
        'displayAll': ''
    }
    response = client.post('/settings', json=data)
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == 'Success'

def test_get_settings(client):
    # Test if getting settings returns a JSON response with status code 200
    response = client.get('/settings')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert 'coloring' in data
    assert 'confidence' in data
    assert 'displayConfidence' in data
    assert 'displayLabel' in data
    assert 'displayAll' in data

def test_start(client):
    # Test if starting the application returns the correct initial step and pieces
    data = {'mode': 'Assembly'}
    response = client.post('/start', json=data)
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert 'step' in data
    assert 'pieces' in data
    assert data['step'] == 1
    assert data['pieces'] == STEPS[1]

def test_next_step(client):
    # Test if going to the next step returns the correct step and pieces
    response = client.post('/next')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert 'step' in data
    assert 'pieces' in data
    assert data['step'] == 2
    assert data['pieces'] == STEPS[2]

def test_previous_step(client):
    # Test if going to the previous step returns the correct step and pieces
    response = client.post('/previous')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert 'step' in data
    assert 'pieces' in data
    assert data['step'] == 1
    assert data['pieces'] == STEPS[1]

def test_send_pieces(client):
    # Test if sending pieces returns a JSON response with status code 200
    data = {'pieces': [LABELS[0], LABELS[1]]}
    response = client.post('/send-pieces', json=data)
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == {'message': 'Necessary pieces sent successfully'}

def test_get_pieces(client):
    # Test if getting pieces returns a JSON response with status code 200
    response = client.get('/send-pieces')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data == [LABELS[0], LABELS[1]]

def test_get_detections_no_results(client):
    # Test if getting detection results when no results are available returns an empty list
    response = client.get('/detections')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data == []

def test_handle_detections(client):
    # Test if sending detection results returns a string response with status code 200
    data = [
        {'label': '0', 'confidence': '0.8', 'boxes': '[10, 20, 100, 150]'},
        {'label': '1', 'confidence': '0.9', 'boxes': '[50, 30, 120, 180]'}
    ]
    response = client.post('/detections', json=data)
    assert response.status_code == 200
    assert response.data == b'Detection results received.'

def test_get_detections(client):
    # Test if getting detection results returns a JSON response with status code 200
    response = client.get('/detections')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data == [
        {'label': '0', 'confidence': '0.8', 'boxes': '[10, 20, 100, 150]'},
        {'label': '1', 'confidence': '0.9', 'boxes': '[50, 30, 120, 180]'}
    ]

def test_handle_labels(client):
    # Test if sending labels returns a JSON response with status code 200
    response = client.post('/labels', json={'step': 1})
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert 'message' in data

def test_invalid_settings(client):
    # Test if sending invalid confidence still returns a JSON response with status code 200
    data = {
        'coloring': 'invalid-coloring',
        'confidence': '101',  # Invalid confidence value
        'displayConfidence': '',
        'displayLabel': '',
        'displayAll': ''
    }
    response = client.post('/settings', json=data)
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == 'Success'

def test_start_invalid_mode(client):
    # Test if starting with an invalid mode returns status code 400
    data = {'mode': 'invalid-mode'}
    response = client.post('/start', json=data)
    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {'error': 'Invalid mode'}

def test_previous_step_reached_minimum(client):
    # Test if going to the previous step when already at the minimum step returns the same step and pieces
    client.post('/previous')
    response = client.post('/previous')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert 'step' in data
    assert 'pieces' in data
    assert data['step'] == 1
    assert data['pieces'] == STEPS[1]

def test_next_step_reached_maximum(client):
    # Test if going to the next step when already at the maximum step returns the same step and pieces
    i=0
    while i < 15:
        client.post('/next')
        i+=1
    response = client.post('/next')
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert 'step' in data
    assert 'pieces' in data
    assert data['step'] == 15
    assert data['pieces'] == STEPS[15]

def test_handle_detections_invalid_data(client):
    # Test if sending invalid detection results data returns a string response with status code 400
    data = [
        {'label': '0', 'confidence': '0.8'},  # Missing 'boxes' key
        {'label': '1', 'confidence': '0.9', 'boxes': '[50, 30, 120, 180]'}
    ]
    response = client.post('/detections', json=data)
    assert response.status_code == 400
    assert response.data == b'Invalid detection results data'

def test_handle_labels_assembly_mode_missing_parts(client):
    # Test handling labels in assembly mode when not all necessary pieces are found
    client.post('/start', json={'mode': 'Assembly'})
    client.post('/send-pieces', json={'pieces': [LABELS[8], LABELS[4], LABELS[15]]})
    data = [
        {'label': '4', 'confidence': '0.8', 'boxes': '[10, 20, 100, 150]'},
        {'label': '8', 'confidence': '0.9', 'boxes': '[50, 30, 120, 180]'},
        # Missing '15' in the detected labels
    ]
    response = client.post('/detections', json=data)
    assert response.status_code == 200
    response = client.post('/labels', json=data)
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data == {'message': 'There is {x} part missing. Check if all pieces are in the view of the camera.'.format(x=1)}

def test_handle_labels_assembly_mode(client):
    # Test handling labels in assembly mode when all necessary pieces are found
    client.post('/start', json={'mode': 'Assembly'})
    client.post('/send-pieces', json={'pieces': [LABELS[8], LABELS[4], LABELS[15]]})
    data = [
        {'label': '4', 'confidence': '0.8', 'boxes': '[10, 20, 100, 150]'},
        {'label': '8', 'confidence': '0.9', 'boxes': '[50, 30, 120, 180]'},
        {'label': '15', 'confidence': '0.7', 'boxes': '[80, 40, 140, 200]'}
    ]
    response = client.post('/detections', json=data)
    assert response.status_code == 200
    response = client.post('/labels', json=data)
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data == {'message': 'All necessary LEGO parts were found. Please grab the marked parts and follow the assembly instructions. Afterwards, press "Next steps" to continue.'}

def test_handle_labels_disassembly_mode_missing_parts(client):
    # Test handling labels in disassembly mode when not all necessary pieces are found
    client.post('/start', json={'mode': 'Disassembly'})
    client.post('/send-pieces', json={'pieces': [LABELS[17]]})
    data = [
        {'label': '4', 'confidence': '0.8', 'boxes': '[10, 20, 100, 150]'}
    ]
    response = client.post('/detections', json=data)
    assert response.status_code == 200
    response = client.post('/labels', json=data)
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data == {'message': 'You did not disassemble the correct parts. Make sure to only disassembly the parts displayed on the screen and place them within the view. There is {x} part missing.'.format(x=1)}

def test_handle_labels_disassembly_mode(client):
    # Test handling labels in disassembly mode when all necessary pieces are found
    client.post('/start', json={'mode': 'Disassembly'})
    client.post('/send-pieces', json={'pieces': [LABELS[17]]})
    data = [
        {'label': '17', 'confidence': '0.8', 'boxes': '[10, 20, 100, 150]'}
    ]
    response = client.post('/detections', json=data)
    assert response.status_code == 200
    response = client.post('/labels', json=data)
    assert response.status_code == 200
    assert response.is_json
    data = response.get_json()
    assert data == {'message': 'All necessary LEGO parts were disassembled correctly. Press "Next Step" to go to the next disassembly step.'}


def test_handle_labels_invalid_step(client):
    # Test handling labels with an invalid step
    response = client.post('/labels', json={'step': 99})
    assert response.status_code == 200