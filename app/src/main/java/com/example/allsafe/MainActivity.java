package com.example.allsafe;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.TextView;
import android.Manifest;

public class MainActivity extends AppCompatActivity {

    private TextView myTextView;
    private Handler handler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        myTextView = findViewById(R.id.textView);
        ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_SMS}, PackageManager.PERMISSION_GRANTED);

        READ_SMS();
    }


    public void READ_SMS(){
        handler.postDelayed(Retrieve_SMS, 1);
    }

    private Runnable Retrieve_SMS = new Runnable() {
        @Override
        public void run() {
            try (Cursor cursor = getContentResolver().query(Uri.parse("content://sms"), null, null, null, null)){
                StringBuilder smsBuilder = new StringBuilder();
                if (cursor != null && cursor.moveToFirst()) {
                    int addressIndex = cursor.getColumnIndex("address");
                    if(addressIndex >= 0){
                        do {
                            String address = cursor.getString(addressIndex);
                            String message = cursor.getString(12);

                            smsBuilder.append("From: ");
                            smsBuilder.append(address);
                            smsBuilder.append("\n");
                            smsBuilder.append("Message:\n");
                            smsBuilder.append(message);
                            smsBuilder.append("\n");
                            smsBuilder.append("______________________________________");
                            smsBuilder.append("\n \n");

                        } while (cursor.moveToNext());
                    }
                }
                myTextView.setText(smsBuilder.toString());
            }
            handler.postDelayed(this, 5000);
        }
    };

}