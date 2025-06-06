package ro.ubb.ai.javaserver.util;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class MeteorologistPasscodeValidator {

    @Value("${app.meteorologist.secret-passcode}")
    private String secretPasscode;

    public boolean isValid(String providedPasscode) {
        return secretPasscode != null && secretPasscode.equals(providedPasscode);
    }
}
