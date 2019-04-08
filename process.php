<?php 

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
   if (isset($_FILES['files'])) {
        /*
          $newDir = True
           if (not $newDir) {
        $path = 'uploads/' . [the dir name that matched]';

        }
        else {
        $uid =  uniqid();
        $path = 'uploads/' . $uid . '_open/';
        //$path = 'uploads/';
        }
        */    
        
        $time_start = microtime(true);

        
        $errors = [];
        $uid =  uniqid();
        $subdir = 'subdir/';
        $father_dir = 'uploads/';
        $uid_path = $uid . '_open/';
        $path = $father_dir . $uid_path;
        //$file = file_get_contents('_open/', FILE_USE_INCLUDE_PATH);
        if(mkdir($path) == false){
            print_r("couldn't create folder @ " . $path);
        }
        @mkdir($path);
        @mkdir($path . $subdir);
        $extensions = ['jpg', 'jpeg', 'png'];
        //echo "<script type=text/javascript'>recieveResults();</script>";

            $file_count = count($_FILES['files']['tmp_name']);

            for ($i = 0; $i < $file_count; $i++) {  
                $file_name = $_FILES['files']['name'][$i];
                $file_tmp = $_FILES['files']['tmp_name'][$i];
                $file_type = $_FILES['files']['type'][$i];
                $file_size = $_FILES['files']['size'][$i];
                $file_ext = strtolower(end(explode('.', $_FILES['files']['name'][$i])));
                $uid1 = uniqid();

                $file = $path . $subdir . $uid1 . '.' . $file_ext;

                if (!in_array($file_ext, $extensions)) {
                    $errors[] = 'Extension not allowed: ' . $file_name . ' ' . $file_type;
                }

                if ($file_size > 2097152) {
                    $errors[] = 'File size exceeds limit: ' . $file_name . ' ' . $file_type;
                }

                if (empty($errors)) {
                    move_uploaded_file($file_tmp, $file);
                    //print 'file saved!';
                }


            }

            $continue = TRUE;
            $contents = glob('uploads/*');
            $is_any = FALSE;
            $tmp_path = $father_dir . $uid . "_closed";
            $i = 0;


            foreach(glob('uploads/*') as $file) {
                $contents[] = $file;
            }

            //echo "<script type=text/javascript'>recieveResults();</script>";
            while ($continue) {
                sleep(1);
                foreach(glob('uploads/*') as $file) {
                    if ($file === $tmp_path) {
                        $is_any = TRUE;
                    }
                    //print "$file";

                    //print "$part $tmp_path <br>";
                }
                //echo "<br>";
                if (!$is_any) {
                    //print 'still waiting';
                } else {
                    $continue = FALSE;
                }
                //echo "<br>";
                $i +=1;
                $is_any = FALSE;
                $time_end = microtime(true);
                $time = $time_end - $time_start;
                if ($time > 25) {
                    exit('timed out, please check if script is running or it has broken');
                }
            }


            $strJsonFileContents = file_get_contents($tmp_path . "/results.json");
            // Convert to array             
            $array = json_decode($strJsonFileContents, true);
            //var_dump($array); // print array

            //var_dump($array["items"]);

            $obj = $array[0];
            
            $time_end = microtime(true);
            $time = $time_end - $time_start;

            
            //if($array["items"]["image_path"] === $uid1) {
                $str = "<h2 style='text-align: center;'>Your Results</h2><img id='resultImg' class='inline' src='" . $array[1] . '/' . $subdir . $uid1 . '.' . $file_ext . "'/><table class='inline' id='resultTable'><tr><td>type</td><td>Result</td><td>Case</td><td>Confidence</td></tr><tr><td>ID</td><td>" . $obj['ID'] . "</td><td>" . $obj['ID-case'] . "</td><td>" . $obj['ID-prob'] . "</td></tr><tr><td>Gender</td><td>" . $obj['GEN'] . "</td><td>" . $obj['GEN-case'] . "</td><td>" . $obj['GEN-prob'] . "</td></tr><tr><td>Age</td><td>" . $obj['AGE'] . "</td><td>" . $obj['AGE-case'] . "</td><td>" . $obj['AGE-prob'] . "</td></tr></table><p>That took $time seconds!</p><p>Disclaimer: These results are the best guesses of the program. We can neither guarantee accuracy nor even a result</p>";
                echo $str;
            //}
            
            if ($errors) {
                print_r($errors);
            }
	}
}   