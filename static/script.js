// Select all list items
const recommendationItems = document.querySelectorAll('.recommendation-item');

// Add a click event listener to each item
recommendationItems.forEach(item => {
    item.addEventListener('click', () => {
        const trackName = item.getAttribute('data-track-name');
        const artists = item.getAttribute('data-artists');
        
         // Construct the Spotify search URL
         const spotifySearchUrl = `https://open.spotify.com/search/${encodeURIComponent(trackName)}`;

         // Open the URL in a new tab
         window.open(spotifySearchUrl, '_blank');
    });
});
