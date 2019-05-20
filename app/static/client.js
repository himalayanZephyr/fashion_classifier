var el = x => document.getElementById(x);

function showPicker(inputId) { el('file-input').click(); }

function showPicked(input) {
    el('upload-label').innerHTML = input.files[0].name;
    var reader = new FileReader();
    reader.onload = function (e) {
        el('image-picked').src = e.target.result;
        el('image-picked').className = '';
    }
    reader.readAsDataURL(input.files[0]);
}

function analyze() {
    var uploadFiles = el('file-input').files;
    if (uploadFiles.length != 1) {
	    alert('Please select 1 file to analyze!');
	    el('analyze-button').innerHTML = 'Analyze';
    }

    el('analyze-button').innerHTML = 'Analyzing...';
    var xhr = new XMLHttpRequest();
    var loc = window.location
    xhr.open('POST', `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`, true);
    xhr.onerror = function() {alert (xhr.responseText);}
    xhr.onload = function(e) {
        if (this.readyState === 4) {
            var response = JSON.parse(e.target.responseText);
            el('result-label').innerHTML = `Supervised Prediction : ${response['result1']}`;
	    el('image-result').src = './static/images/cam_supervised.jpg?rand='+Math.floor(Math.random() * 10000) + 1; 
	    el('image-result').className = '';
  	    el('result-label2').innerHTML = `Unsupervised Prediction : ${response['result2']}`;
            el('image-result2').src = './static/images/cam_rotation.jpg?rand='+Math.floor(Math.random() * 10000) + 10000;
            el('image-result2').className = '';
	
        }
        el('analyze-button').innerHTML = 'Analyze';
    }

    var fileData = new FormData();
    fileData.append('file', uploadFiles[0]);
    xhr.send(fileData);
}

