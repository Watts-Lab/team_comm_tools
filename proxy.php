<?php
$url = $_GET['url'];

$options = [
    'http' => [
        'follow_location' => 1,
        'max_redirects' => 10,
    ],
];

$context = stream_context_create($options);

$content = file_get_contents($url, false, $context);

if ($content === false) {
    http_response_code(500);
    echo "Failed to fetch content.";
} else {
    header('Content-Type: text/html');
    echo $content;
}
?>