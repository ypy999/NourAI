package ai.jebrex.app

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

/**
 * Minimal native Android client for Jebrex AI.
 * Talks exclusively to the local/LAN Jebrex AI core backend over HTTPS.
 */
class MainActivity : AppCompatActivity() {
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var apiClient: JebrexApiClient
    private lateinit var transcriptView: TextView
    private lateinit var inputField: EditText
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val defaultBaseUrl = "https://10.0.2.2:7860"
        apiClient = JebrexApiClient(defaultBaseUrl)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 32, 24, 24)
        }

        statusView = TextView(this).apply {
            text = "Jebrex AI Android — Independent Local Core ($defaultBaseUrl)"
            textSize = 14f
        }
        transcriptView = TextView(this).apply {
            text = "Ready. Messages go directly to your Jebrex AI neural backend.\n\n"
            textSize = 15f
        }
        val scroll = ScrollView(this).apply {
            addView(transcriptView)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
            )
        }

        val inputRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        inputField = EditText(this).apply {
            hint = "اسأل Jebrex AI / Ask Jebrex AI..."
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val sendBtn = Button(this).apply {
            text = "Send"
            setOnClickListener { sendCurrentMessage() }
        }
        inputRow.addView(inputField)
        inputRow.addView(sendBtn)

        root.addView(statusView)
        root.addView(scroll)
        root.addView(inputRow)
        setContentView(root)
    }

    private fun sendCurrentMessage() {
        val msg = inputField.text.toString().trim()
        if (msg.isEmpty()) return
        inputField.setText("")
        transcriptView.append("[YOU]\n$msg\n\n")
        statusView.text = "Generating from Jebrex AI core..."

        executor.execute {
            try {
                val res = apiClient.sendMessage(msg, language = "arabic")
                runOnUiThread {
                    transcriptView.append(
                        "[JEBREX AI | ${res.responseSource} | ${res.generatedTokens} tok]\n${res.reply}\n\n"
                    )
                    statusView.text = "Connected | ${res.responseSource}"
                }
            } catch (e: Exception) {
                runOnUiThread {
                    transcriptView.append("[ERROR]\n${e.message}\n\n")
                    statusView.text = "Backend error"
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdownNow()
    }
}
