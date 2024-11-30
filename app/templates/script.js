const dropArea = document.querySelector('.drag-area');

// Prevent default behaviors when dragging
dropArea.addEventListener('dragover', (event) => {
    event.preventDefault(); // Prevent the default drag behaviors
    event.dataTransfer.dropEffect = 'copy'; // Indicate what kind of drop is allowed
});

// Handle the drop event
dropArea.addEventListener('drop', (event) => {
    event.preventDefault(); // Prevent default behavior (open file)
    const file = event.dataTransfer.files[0];
    if (file) {
        sendFileToServer(file); // Send the file to the server
    }
});

// Trigger file input click when the browse button is clicked
document.getElementById('browseButton').addEventListener('click', function() {
    document.getElementById('fileInput').click(); // Trigger file input
});

// Handle file selection from the file input
document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        sendFileToServer(file); // Send the file to the server
    }
});

// Function to send the file to the server
function sendFileToServer(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log(data); // Handle the response
        if (data.urls) {
            displayUrls(data.urls); // Display URLs from server response
        }
    })
    .catch(error => console.error('Error:', error));
}

// Function to display URLs on the page
function displayUrls(urls) {
    const urlList = document.getElementById('urlList');
    urlList.innerHTML = ''; // Clear previous URLs

    urls.forEach(url => {
        const linkElement = document.createElement('a');
        linkElement.href = url;
        linkElement.target = '_blank'; // Open in a new tab
        linkElement.textContent = url; // Set the text to the URL
        urlList.appendChild(linkElement);
        urlList.appendChild(document.createElement('br')); // Add a line break
    });
}
