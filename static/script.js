document.addEventListener('DOMContentLoaded', function() {
    // Event listener for 'Fetch Data' button
    document.getElementById('fetchDataButton').addEventListener('click', function() {
        console.log('Fetch Data Button Clicked');  // Debugging log
        fetch('/api')
            .then(response => response.json())
            .then(data => {
                document.getElementById('apiMessage').innerText = data.message;
            });
    });

    // Function to generate a random message
    function getRandomMessage() {
        const messages = [
            "Keep up the great work!",
            "You're doing awesome!",
            "Flask is amazing!",
            "JavaScript is fun!",
            "You are learning fast!"
        ];
        const randomIndex = Math.floor(Math.random() * messages.length);
        return messages[randomIndex];
    }

    // Event listener for 'Random' button
    document.getElementById('randomButton').addEventListener('click', function() {
        console.log('Random Button Clicked');  // Debugging log
        document.getElementById('apiMessage').innerText = getRandomMessage();
    });
});
